"""
Reusable data pipeline for the World Cup 2026 predictor.

This module builds a single, deduplicated "master" dataset of international
football matches by combining the Kaggle historical CSVs with completed
national-team fixtures fetched from football-data.org (v4). The result is
written to both a CSV and a SQLite database so that every model notebook and
the prediction notebook can load from one canonical source via ``load_from_db``.

It is importable from notebooks/scripts AND runnable as a CLI:

    python scripts/data_pipeline.py

The CLI reads ``FOOTBALL_DATA_API_KEY`` from the environment (optionally from a
``.env`` file). With no key it builds a historical-only dataset; with a key it
appends new matches fetched from the API. The API key is NEVER hardcoded.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# pandas >= 3.0 infers Arrow-backed string columns by default, which breaks
# scikit-learn fold indexing in the model notebooks. Revert to object dtype so
# behaviour matches the environment the notebooks were authored in.
try:
    pd.set_option("future.infer_string", False)
except (KeyError, ValueError):
    pass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
TOURNAMENT_DIR = DATA_DIR / "tournament"
LIVE_DIR = DATA_DIR / "live"
COMBINED_DIR = DATA_DIR / "combined"
LOGS_DIR = DATA_DIR / "logs"

HISTORICAL_RESULTS_CSV = HISTORICAL_DIR / "results.csv"
HISTORICAL_SHOOTOUTS_CSV = HISTORICAL_DIR / "shootouts.csv"

API_MATCHES_CSV = LIVE_DIR / "api_matches.csv"
MASTER_CSV = COMBINED_DIR / "master_matches.csv"
MASTER_DB = COMBINED_DIR / "master_matches.db"
UPDATE_LOG = LOGS_DIR / "update_log.json"


# ---------------------------------------------------------------------------
# Master schema (STEP 4)
# ---------------------------------------------------------------------------

MASTER_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
    "is_shootout",
    "shootout_winner",
    "source",
]


# ---------------------------------------------------------------------------
# Team-name canonicalisation (STEP 3d)
# ---------------------------------------------------------------------------

# Canonical name map. Keys are alternative spellings that appear in either the
# Kaggle dataset, the WC 2026 team list, or the football-data.org API; values
# are the canonical names used throughout the project (matching the Kaggle
# `results.csv` spellings).
TEAM_NAME_MAP: dict[str, str] = {
    # Covered already across model.ipynb / lr_model.ipynb / streamlit_app.py
    "USA": "United States",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Serbia and Montenegro": "Serbia",  # historical alias
    "China": "China PR",                 # historical alias
    "Czechia": "Czech Republic",
    # Common football-data.org -> Kaggle spelling mismatches
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Czech Republic ": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Republic of Ireland": "Ireland",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Cabo Verde Islands": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "North Macedonia": "North Macedonia",
    "United States of America": "United States",
    "USMNT": "United States",
}


def standardise_team_name(name: str) -> str:
    """Return the canonical team name for ``name``.

    Performs Unicode (NFC) normalisation, trims surrounding whitespace, then
    applies :data:`TEAM_NAME_MAP`. Names not present in the map are returned
    unchanged (after normalisation).
    """
    if name is None:
        return name
    normalised = unicodedata.normalize("NFC", str(name)).strip()
    return TEAM_NAME_MAP.get(normalised, normalised)


# ---------------------------------------------------------------------------
# 3a. load_historical
# ---------------------------------------------------------------------------

def load_historical() -> pd.DataFrame:
    """Load the Kaggle historical CSVs into the master schema.

    Reads ``data/historical/results.csv`` and ``data/historical/shootouts.csv``,
    standardises team names, derives ``is_shootout`` / ``shootout_winner`` from
    the shootouts file, tags ``source = "historical_csv"`` and returns a single
    DataFrame with exactly :data:`MASTER_COLUMNS`.
    """
    results = pd.read_csv(HISTORICAL_RESULTS_CSV, dtype={"date": str})
    shootouts = pd.read_csv(HISTORICAL_SHOOTOUTS_CSV, dtype={"date": str})

    results["home_team"] = results["home_team"].map(standardise_team_name)
    results["away_team"] = results["away_team"].map(standardise_team_name)

    shootouts["home_team"] = shootouts["home_team"].map(standardise_team_name)
    shootouts["away_team"] = shootouts["away_team"].map(standardise_team_name)
    shootouts["winner"] = shootouts["winner"].map(standardise_team_name)

    # Map shootout winner by (date, home_team, away_team)
    shootout_winner_lookup: dict[tuple[str, str, str], str] = {}
    for _, row in shootouts.iterrows():
        key = (str(row["date"]), row["home_team"], row["away_team"])
        shootout_winner_lookup[key] = row["winner"]

    def _shootout_winner(row) -> str | None:
        return shootout_winner_lookup.get(
            (str(row["date"]), row["home_team"], row["away_team"])
        )

    df = pd.DataFrame()
    df["date"] = results["date"].astype(str)
    df["home_team"] = results["home_team"]
    df["away_team"] = results["away_team"]
    df["home_score"] = pd.to_numeric(results["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(results["away_score"], errors="coerce").astype("Int64")
    df["tournament"] = results["tournament"]
    df["neutral"] = results["neutral"].astype(str).str.upper() == "TRUE"
    df["shootout_winner"] = results.apply(_shootout_winner, axis=1)
    df["is_shootout"] = df["shootout_winner"].notna()
    df["source"] = "historical_csv"

    return df[MASTER_COLUMNS].copy()


# ---------------------------------------------------------------------------
# 3b. fetch_api_matches
# ---------------------------------------------------------------------------

API_BASE_URL = "https://api.football-data.org/v4"

# Logical tournament name -> football-data.org competition code. Not every code
# is available on every subscription tier; unavailable ones are skipped with a
# warning so the pipeline degrades gracefully.
COMPETITION_CODES: dict[str, str] = {
    "FIFA World Cup": "WC",
    "World Cup Qualifiers": "WCQ",
    "UEFA Nations League": "UNL",
    "Copa America": "COPA",
    "UEFA European Championship": "EC",
    "Africa Cup of Nations": "AFCN",
    "AFC Asian Cup": "AFC",
    "CONCACAF Gold Cup": "GC",
    "Friendly": "FR",
}

_REQUEST_PAUSE_SECONDS = 6  # be polite to the free-tier rate limit (10 req/min)


def _api_result_to_rows(matches: list[dict], tournament: str) -> list[dict]:
    rows: list[dict] = []
    for m in matches:
        if m.get("status") != "FINISHED":
            continue

        score = m.get("score", {}) or {}
        full_time = score.get("fullTime", {}) or {}
        home_score = full_time.get("home")
        away_score = full_time.get("away")
        if home_score is None or away_score is None:
            continue

        home_team = standardise_team_name(m.get("homeTeam", {}).get("name", ""))
        away_team = standardise_team_name(m.get("awayTeam", {}).get("name", ""))
        if not home_team or not away_team:
            continue

        penalties = score.get("penalties", {}) or {}
        duration = score.get("duration")
        is_shootout = (
            duration == "PENALTY_SHOOTOUT"
            or penalties.get("home") is not None
            or penalties.get("away") is not None
        )
        shootout_winner = None
        if is_shootout:
            winner_flag = score.get("winner")
            if winner_flag == "HOME_TEAM":
                shootout_winner = home_team
            elif winner_flag == "AWAY_TEAM":
                shootout_winner = away_team

        rows.append(
            {
                "date": str(m.get("utcDate", ""))[:10],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "tournament": tournament,
                "neutral": False,
                "is_shootout": bool(is_shootout),
                "shootout_winner": shootout_winner,
                "source": "football_data_org",
            }
        )
    return rows


def _fetch_competition(
    api_key: str, code: str, tournament: str, since_date: str | None
) -> list[dict]:
    """Fetch finished matches for one competition.

    Returns a list of master-schema row dicts. Returns an empty list (with a
    warning) when the competition endpoint is unavailable for this API key.
    Raises ``RuntimeError`` on authentication / unexpected HTTP failures.
    """
    url = f"{API_BASE_URL}/competitions/{code}/matches"
    params: dict[str, str] = {"status": "FINISHED"}
    today_utc = datetime.now(timezone.utc).date()
    if since_date:
        # football-data.org rejects requests when dateFrom is not before dateTo.
        parsed_since = datetime.strptime(since_date, "%Y-%m-%d").date()
        date_to = today_utc
        if parsed_since >= date_to:
            # If update log stores a future/today date, query a safe 1-day window.
            adjusted_from = (date_to - timedelta(days=1)).strftime("%Y-%m-%d")
            params["dateFrom"] = adjusted_from
            params["dateTo"] = date_to.strftime("%Y-%m-%d")
        else:
            params["dateFrom"] = since_date
            # football-data.org requires dateTo when dateFrom is set
            params["dateTo"] = date_to.strftime("%Y-%m-%d")

    try:
        response = requests.get(
            url,
            headers={"X-Auth-Token": api_key},
            params=params,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error contacting football-data.org: {exc}") from exc

    if response.status_code in (403, 404):
        warnings.warn(
            f"Competition '{tournament}' ({code}) unavailable for this API key "
            f"(HTTP {response.status_code}); skipping.",
            stacklevel=2,
        )
        return []
    if response.status_code == 429:
        warnings.warn(
            f"Rate limited while fetching '{tournament}' ({code}); skipping.",
            stacklevel=2,
        )
        return []
    if response.status_code != 200:
        raise RuntimeError(
            f"football-data.org returned HTTP {response.status_code} for "
            f"'{tournament}' ({code}): {response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Could not parse API response for '{tournament}': {exc}") from exc

    return _api_result_to_rows(payload.get("matches", []), tournament)


def fetch_api_matches(api_key: str, since_date: str | None = None) -> pd.DataFrame:
    """Fetch completed national-team fixtures from football-data.org (v4).

    Iterates over the competition categories in :data:`COMPETITION_CODES`,
    applies ``since_date`` when provided, standardises team names, and returns a
    DataFrame in the master schema with ``source = "football_data_org"``.

    Raises ``RuntimeError`` on HTTP/network errors; logs a warning and skips a
    competition whose endpoint is unavailable for the supplied key.
    """
    if not api_key:
        raise RuntimeError("fetch_api_matches requires a non-empty api_key.")

    all_rows: list[dict] = []
    competitions = list(COMPETITION_CODES.items())
    for idx, (tournament, code) in enumerate(competitions):
        all_rows.extend(_fetch_competition(api_key, code, tournament, since_date))
        if idx < len(competitions) - 1:
            time.sleep(_REQUEST_PAUSE_SECONDS)

    if not all_rows:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    df = pd.DataFrame(all_rows)
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    return df[MASTER_COLUMNS].copy()


# ---------------------------------------------------------------------------
# 3e. merge_and_deduplicate
# ---------------------------------------------------------------------------

def merge_and_deduplicate(hist: pd.DataFrame, api: pd.DataFrame) -> pd.DataFrame:
    """Concatenate historical + API matches and drop duplicates.

    Deduplicates on ``(date, home_team, away_team)`` keeping the
    ``historical_csv`` row when the same match exists in both sources, then
    sorts by date ascending.
    """
    if api is None or api.empty:
        combined = hist.copy()
    else:
        combined = pd.concat([hist, api], ignore_index=True)

    # Historical first so keep="first" prefers it on duplicates.
    source_rank = combined["source"].map({"historical_csv": 0, "football_data_org": 1})
    combined = combined.assign(_rank=source_rank.fillna(2))
    combined = combined.sort_values(["date", "home_team", "away_team", "_rank"])
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="first"
    )
    combined = combined.drop(columns="_rank")
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined[MASTER_COLUMNS].copy()


# ---------------------------------------------------------------------------
# 3f. write_update_log
# ---------------------------------------------------------------------------

def write_update_log(
    new_count: int, last_date: str, total: int, n_hist: int, n_api: int
) -> None:
    """Append a JSON record describing this update run to the update log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "new_matches_added": int(new_count),
        "last_match_date": last_date,
        "total_matches": int(total),
        "n_historical": int(n_hist),
        "n_api": int(n_api),
    }
    with open(UPDATE_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _read_last_update_date() -> str | None:
    """Return the ``last_match_date`` from the most recent log record, if valid."""
    if not UPDATE_LOG.exists():
        return None
    try:
        with open(UPDATE_LOG, encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return None
    if not lines:
        return None
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    candidate = last.get("last_match_date")
    if not candidate or candidate == "N/A":
        return None
    try:
        datetime.strptime(candidate, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    return candidate


# ---------------------------------------------------------------------------
# STEP 5. SQLite helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT NOT NULL,
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    home_score       INTEGER,
    away_score       INTEGER,
    tournament       TEXT,
    neutral          INTEGER,
    is_shootout      INTEGER,
    shootout_winner  TEXT,
    source           TEXT,
    UNIQUE(date, home_team, away_team)
);
"""


def _write_sqlite(df: pd.DataFrame) -> None:
    """(Over)write the SQLite ``matches`` table from ``df``."""
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MASTER_DB)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS matches;")
        cur.executescript(_CREATE_TABLE_SQL)

        def _int_or_none(value):
            return None if pd.isna(value) else int(value)

        rows = [
            (
                str(r.date),
                r.home_team,
                r.away_team,
                _int_or_none(r.home_score),
                _int_or_none(r.away_score),
                None if pd.isna(r.tournament) else str(r.tournament),
                1 if bool(r.neutral) else 0,
                1 if bool(r.is_shootout) else 0,
                None if pd.isna(r.shootout_winner) else str(r.shootout_winner),
                r.source,
            )
            for r in df.itertuples(index=False)
        ]
        cur.executemany(
            """
            INSERT OR IGNORE INTO matches
                (date, home_team, away_team, home_score, away_score,
                 tournament, neutral, is_shootout, shootout_winner, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def load_from_db() -> pd.DataFrame:
    """Load the master ``matches`` table as a DataFrame in the master schema.

    This is the primary data source for the Dixon-Coles / Poisson models and the
    updated predictions notebook. ``neutral`` and ``is_shootout`` are returned as
    booleans; ``home_score`` / ``away_score`` are nullable integers.
    """
    if not MASTER_DB.exists():
        raise FileNotFoundError(
            f"{MASTER_DB} not found. Run build_master_dataset() first "
            "(e.g. `python scripts/data_pipeline.py`)."
        )
    conn = sqlite3.connect(MASTER_DB)
    try:
        df = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score, "
            "tournament, neutral, is_shootout, shootout_winner, source "
            "FROM matches ORDER BY date;",
            conn,
        )
    finally:
        conn.close()

    # Scores are returned as float64 (NaN for missing/future fixtures) to mirror
    # the dtype the model notebooks previously got from pd.read_csv().
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].fillna(0).astype(int).astype(bool)
    df["is_shootout"] = df["is_shootout"].fillna(0).astype(int).astype(bool)
    return df[MASTER_COLUMNS].copy()


# ---------------------------------------------------------------------------
# 3c. build_master_dataset
# ---------------------------------------------------------------------------

def build_master_dataset(api_key: str | None = None) -> pd.DataFrame:
    """Build, persist and return the deduplicated master match dataset.

    With no ``api_key`` the dataset is historical-only. With a key, new matches
    on/after the last logged match date are fetched and merged. Results are
    written to ``master_matches.csv`` and ``master_matches.db`` and a summary is
    printed to stdout.
    """
    hist = load_historical()
    n_hist = len(hist)
    n_api = 0
    api = pd.DataFrame(columns=MASTER_COLUMNS)
    api_last_date = None

    if api_key:
        since_date = _read_last_update_date()
        api = fetch_api_matches(api_key, since_date=since_date)
        n_api = len(api)
        if not api.empty:
            api_last_date = str(api["date"].max())
        # Persist the raw API pull for transparency/debugging.
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        api.to_csv(API_MATCHES_CSV, index=False)

    master = merge_and_deduplicate(hist, api)
    total = len(master)
    new_count = max(0, total - n_hist)
    master_last_date = str(master["date"].max()) if total else "N/A"

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_CSV, index=False)
    _write_sqlite(master)

    write_update_log(
        new_count=new_count,
        last_date=master_last_date,
        total=total,
        n_hist=n_hist,
        n_api=n_api,
    )

    print(f"New matches added:        {new_count}")
    print(f"Last successful API date: {api_last_date or 'N/A'}")
    print(f"Total training matches:   {total}")
    print(f"  Historical CSV:         {n_hist}")
    print(f"  API-added:              {n_api}")

    return master


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        print(
            "WARNING: FOOTBALL_DATA_API_KEY not set; building historical-only "
            "dataset (no API matches will be fetched)."
        )
        build_master_dataset()
    else:
        build_master_dataset(api_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
