"""
Fetch finished World Cup 2026 matches from football-data.org and update data/live/wc2026_results.csv.

Usage:
    FOOTBALL_DATA_API_KEY=<token> python scripts/fetch_results.py

Exits with code 0 on success (even if no new results), code 1 on API failure.
"""

import os
import sys
import difflib
from pathlib import Path
from datetime import timezone

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHES_CSV = REPO_ROOT / "data" / "tournament" / "matches.csv"
RESULTS_CSV = REPO_ROOT / "data" / "live" / "wc2026_results.csv"
TEAMS_CSV = REPO_ROOT / "data" / "tournament" / "teams.csv"

API_URL = "https://api.football-data.org/v4/competitions/WC/matches"
API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
DEBUG_LOG = REPO_ROOT / "debug-685eb9.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    import json
    import time

    payload = {
        "sessionId": "685eb9",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # #endregion


def fuzzy_match_team(api_name: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    """Return the best fuzzy match for api_name in candidates, or None."""
    matches = difflib.get_close_matches(api_name, candidates, n=1, cutoff=threshold)
    if matches:
        return matches[0]
    # Fall back to case-insensitive substring search
    api_lower = api_name.lower()
    for candidate in candidates:
        if api_lower in candidate.lower() or candidate.lower() in api_lower:
            return candidate
    return None


def determine_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    elif away_score > home_score:
        return "A"
    return "D"


def main() -> int:
    _agent_log(
        "B",
        "fetch_results.py:main:entry",
        "fetch_results started",
        {"api_key_present": bool(API_KEY), "api_key_len": len(API_KEY)},
    )
    if not API_KEY:
        print("ERROR: FOOTBALL_DATA_API_KEY environment variable is not set.")
        _agent_log("B", "fetch_results.py:main:no_key", "missing API key", {"exit_code": 1})
        return 1

    # --- Load local data ---
    try:
        matches_df = pd.read_csv(MATCHES_CSV)
        teams_df = pd.read_csv(TEAMS_CSV)
    except FileNotFoundError as e:
        print(f"ERROR: Could not load required CSV: {e}")
        return 1

    try:
        results_df = pd.read_csv(RESULTS_CSV)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=["match_id", "home_score", "away_score", "result", "played_at"])

    # Build a map: team_id -> team_name
    team_id_to_name: dict[int, str] = dict(zip(teams_df["id"], teams_df["team_name"]))
    team_names: list[str] = teams_df["team_name"].tolist()

    # Build a lookup: (home_team_name, away_team_name) -> match_id
    match_lookup: dict[tuple[str, str], int] = {}
    for _, row in matches_df.iterrows():
        home_name = team_id_to_name.get(row["home_team_id"])
        away_name = team_id_to_name.get(row["away_team_id"])
        if home_name and away_name:
            match_lookup[(home_name, away_name)] = int(row["id"])

    already_recorded: set[int] = set(results_df["match_id"].astype(int).tolist())

    # --- Fetch from API ---
    try:
        response = requests.get(
            API_URL,
            headers={"X-Auth-Token": API_KEY},
            params={"status": "FINISHED"},
            timeout=30,
        )
        response.raise_for_status()
        api_data = response.json()
    except requests.RequestException as e:
        print(f"ERROR: API request failed: {e}")
        _agent_log(
            "C",
            "fetch_results.py:main:api_error",
            "API request failed",
            {"error": str(e), "exit_code": 1},
        )
        return 1
    except ValueError as e:
        print(f"ERROR: Failed to parse API response: {e}")
        _agent_log(
            "C",
            "fetch_results.py:main:parse_error",
            "API response parse failed",
            {"error": str(e), "exit_code": 1},
        )
        return 1

    _agent_log(
        "C",
        "fetch_results.py:main:api_ok",
        "API request succeeded",
        {"status_code": response.status_code, "match_count": len(api_data.get("matches", []))},
    )

    api_matches = api_data.get("matches", [])

    new_rows: list[dict] = []

    for api_match in api_matches:
        if api_match.get("status") != "FINISHED":
            continue

        score = api_match.get("score", {})
        full_time = score.get("fullTime", {})
        home_score = full_time.get("home")
        away_score = full_time.get("away")

        if home_score is None or away_score is None:
            continue

        api_home = api_match.get("homeTeam", {}).get("name", "")
        api_away = api_match.get("awayTeam", {}).get("name", "")

        matched_home = fuzzy_match_team(api_home, team_names)
        matched_away = fuzzy_match_team(api_away, team_names)

        if not matched_home or not matched_away:
            print(f"WARNING: Could not match teams: '{api_home}' / '{api_away}' — skipping.")
            continue

        match_id = match_lookup.get((matched_home, matched_away))
        if match_id is None:
            print(f"WARNING: No match found for {matched_home} vs {matched_away} — skipping.")
            continue

        if match_id in already_recorded:
            continue

        utc_date = api_match.get("utcDate", "")
        result = determine_result(int(home_score), int(away_score))

        new_rows.append({
            "match_id": match_id,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "result": result,
            "played_at": utc_date,
        })
        already_recorded.add(match_id)

    _agent_log(
        "F",
        "fetch_results.py:main:processed",
        "finished processing API matches",
        {
            "api_finished_matches": len(api_matches),
            "new_rows": len(new_rows),
            "already_recorded": len(already_recorded),
        },
    )

    if not new_rows:
        print("No new results")
        _agent_log("F", "fetch_results.py:main:no_new", "no new results to write", {"exit_code": 0})
        return 0

    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([results_df, new_df], ignore_index=True)
    updated_df.to_csv(RESULTS_CSV, index=False)
    print(f"Updated results.csv with {len(new_rows)} new result{'s' if len(new_rows) != 1 else ''}")
    _agent_log(
        "E",
        "fetch_results.py:main:written",
        "wrote updated results csv",
        {"new_rows": len(new_rows), "total_rows": len(updated_df), "exit_code": 0},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
