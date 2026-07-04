"""
Evaluate WC 2026 model accuracy against actual tournament results.

Used by predictions/wc2026_accuracy.ipynb. Compares each model's win/draw/loss
predictions (and scorelines for Poisson / Dixon-Coles) to finished matches,
broken down by tournament stage.

Group-stage accuracy uses direct predict_group_match() calls (pre-match style).
Knockout accuracy replays the full bracket with actual results frozen through the
previous stage — the same logic as updated_predictions.ipynb with
ACTUALS_THROUGH_STAGE_ID set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from scripts.data_pipeline import standardise_team_name

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "tournament"
LIVE = REPO_ROOT / "data" / "live"

# All models available under models/*.ipynb
MODELS: dict[str, str] = {
    "elo": "ELO",
    "lr_model": "LR",
    "ensemble_model": "Ensemble",
    "poisson_model": "Poisson",
    "dc_model": "Dixon-Coles",
}

# Models that predict exact scorelines (used for score accuracy)
SCORELINE_MODELS = {"poisson_model", "dc_model"}

# stage_id -> display name (matches tournament_stages.csv + knockout sim labels)
STAGE_ID_TO_NAME: dict[int, str] = {
    1: "Group Stage",
    2: "Round of 32",
    3: "Round of 16",
    4: "Quarterfinals",
    5: "Semifinals",
    6: "Third Place Playoff",
    7: "Final",
}

# Knockout simulation uses "Bronze Final" for stage 6 — map back to CSV name
KO_STAGE_ALIASES: dict[str, str] = {
    "Bronze Final": "Third Place Playoff",
}

PredictGroupFn = Callable[[str, str], tuple[int, int, str | None]]
PredictKnockoutFn = Callable[[str, str], tuple[str, str, float, float]]
GetEloFn = Callable[[str], float]


@dataclass
class StageAccuracy:
    """Accuracy for one model on one tournament stage."""

    stage_id: int
    stage_name: str
    model_key: str
    model_label: str
    n_matches: int
    n_correct: int
    win_accuracy: float
    n_score_matches: int = 0
    n_score_correct: int = 0
    score_accuracy: float | None = None


@dataclass
class AccuracyReport:
    """Full accuracy report for one model."""

    model_key: str
    model_label: str
    stages: list[StageAccuracy] = field(default_factory=list)


def repo_root() -> Path:
    return REPO_ROOT


def load_tournament_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load schedule, teams, stages, and committed live results."""
    matches = pd.read_csv(DATA / "matches.csv")
    teams = pd.read_csv(DATA / "teams.csv")
    stages = pd.read_csv(DATA / "tournament_stages.csv")
    results = pd.read_csv(LIVE / "wc2026_results.csv")
    return matches, teams, stages, results


def build_team_lookups(teams_df: pd.DataFrame) -> tuple[dict[int, str], dict[str, str], dict[str, str]]:
    """Build id->name, name->group, and canonical->csv name maps."""
    id_to_name = dict(zip(teams_df["id"], teams_df["team_name"]))
    name_to_group = dict(zip(teams_df["team_name"], teams_df["group_letter"]))
    canonical_to_csv = {standardise_team_name(name): name for name in id_to_name.values()}
    return id_to_name, name_to_group, canonical_to_csv


def load_completed_from_csv(
    results_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    id_to_name: dict[int, str],
) -> dict[frozenset, dict]:
    """
    Build wc2026_completed dict from committed wc2026_results.csv.

    Keys are frozenset({team_a, team_b}); values hold home/away scores and winner.
    Winner uses regulation-time result codes (H/D/A).
    """
    completed: dict[frozenset, dict] = {}
    merged = results_df.merge(matches_df, left_on="match_id", right_on="id")

    for _, row in merged.iterrows():
        home = id_to_name[int(row["home_team_id"])]
        away = id_to_name[int(row["away_team_id"])]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        if row["result"] == "H":
            winner = home
        elif row["result"] == "A":
            winner = away
        else:
            winner = None  # draw
        completed[frozenset({home, away})] = {
            "home": home,
            "away": away,
            "home_score": hs,
            "away_score": as_,
            "winner": winner,
            "match_id": int(row["match_id"]),
            "stage_id": int(row["stage_id"]),
        }
    return completed


def supplement_from_api(
    completed: dict[frozenset, dict],
    canonical_to_csv: dict[str, str],
    api_key: str,
) -> dict[frozenset, dict]:
    """Merge any extra FINISHED matches from the API (overwrites CSV rows)."""
    if not api_key:
        return completed

    import requests

    resp = requests.get(
        "https://api.football-data.org/v4/competitions/WC/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "FINISHED"},
        timeout=30,
    )
    if resp.status_code != 200:
        return completed

    for m in resp.json().get("matches", []):
        canon_ht = standardise_team_name(m["homeTeam"]["name"])
        canon_at = standardise_team_name(m["awayTeam"]["name"])
        ht = canonical_to_csv.get(canon_ht)
        at = canonical_to_csv.get(canon_at)
        if not ht or not at:
            continue
        hs = m["score"]["fullTime"]["home"]
        as_ = m["score"]["fullTime"]["away"]
        if hs is None or as_ is None:
            continue
        raw_winner = m["score"].get("winner")
        winner = ht if raw_winner == "HOME_TEAM" else (at if raw_winner == "AWAY_TEAM" else None)
        completed[frozenset({ht, at})] = {
            "home": ht,
            "away": at,
            "home_score": int(hs),
            "away_score": int(as_),
            "winner": winner,
        }
    return completed


def outcome_from_scores(home_goals: int, away_goals: int) -> str:
    """Return H/D/A from a scoreline."""
    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def build_standings(gs_df: pd.DataFrame, name_to_group: dict[str, str], get_elo: GetEloFn) -> pd.DataFrame:
    """Build group standings with FIFA tiebreakers (same as updated_predictions.ipynb)."""
    stats: dict[str, dict] = {}
    h2h: dict[tuple, int] = {}

    for _, r in gs_df.iterrows():
        h, a = r["home"], r["away"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        if hg > ag:
            h2h[(h, a)], h2h[(a, h)] = 3, 0
        elif hg == ag:
            h2h[(h, a)], h2h[(a, h)] = 1, 1
        else:
            h2h[(h, a)], h2h[(a, h)] = 0, 3
        for team, gf, ga in [
            (r["home"], r["home_goals"], r["away_goals"]),
            (r["away"], r["away_goals"], r["home_goals"]),
        ]:
            if team not in stats:
                stats[team] = {"gp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0}
            s = stats[team]
            s["gp"] += 1
            s["gf"] += gf
            s["ga"] += ga
            if gf > ga:
                s["w"] += 1
                s["pts"] += 3
            elif gf == ga:
                s["d"] += 1
                s["pts"] += 1
            else:
                s["l"] += 1

    rows = []
    for team, s in stats.items():
        rows.append({
            "team": team,
            "group": name_to_group[team],
            "gp": s["gp"],
            "w": s["w"],
            "d": s["d"],
            "l": s["l"],
            "gf": s["gf"],
            "ga": s["ga"],
            "gd": s["gf"] - s["ga"],
            "pts": s["pts"],
            "elo": get_elo(team),
        })

    df = pd.DataFrame(rows)

    def calc_h2h_pts(team, opponents):
        return sum(h2h.get((team, opp), 0) for opp in opponents)

    ranked = []
    for _, grp in df.groupby("group"):
        grp = grp.sort_values("pts", ascending=False).reset_index(drop=True)
        tiers = []
        for _, tied in grp.groupby("pts", sort=False):
            if len(tied) == 1:
                tiers.append(tied)
            else:
                tied_teams = tied["team"].tolist()
                tied = tied.copy()
                tied["h2h_pts"] = tied["team"].apply(
                    lambda t: calc_h2h_pts(t, [x for x in tied_teams if x != t])
                )
                tied = tied.sort_values(
                    ["h2h_pts", "gd", "gf", "elo"],
                    ascending=[False, False, False, False],
                ).drop(columns="h2h_pts")
                tiers.append(tied)
        ranked.append(pd.concat(tiers))

    df = pd.concat(ranked).reset_index(drop=True)
    df["rank"] = df.groupby("group").cumcount() + 1
    return df


def simulate_tournament(
    matches_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
    id_to_name: dict[int, str],
    name_to_group: dict[str, str],
    completed: dict[frozenset, dict],
    predict_group: PredictGroupFn,
    predict_knockout: PredictKnockoutFn,
    get_elo: GetEloFn,
    actuals_through_stage_id: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full tournament simulation (group + knockout).

  If actuals_through_stage_id is set, only results through that stage are treated
  as actual; later stages are always model-predicted.
    """
    # --- Group stage ---
    gs_results = []
    for _, row in matches_df[matches_df["stage_id"] == 1].iterrows():
        home = id_to_name[int(row["home_team_id"])]
        away = id_to_name[int(row["away_team_id"])]
        actual = completed.get(frozenset({home, away}))
        if actuals_through_stage_id is not None and int(row["stage_id"]) > actuals_through_stage_id:
            actual = None
        if actual:
            hg, ag = actual["home_score"], actual["away_score"]
            if actual["home"] != home:
                hg, ag = ag, hg
            winner = home if hg > ag else (away if ag > hg else "Draw")
            source = "actual"
        else:
            hg, ag, w = predict_group(home, away)
            winner = w if w else "Draw"
            source = "predicted"
        gs_results.append({
            "match": int(row["match_number"]),
            "match_id": int(row["id"]),
            "stage_id": 1,
            "group": row["match_label"],
            "home": home,
            "away": away,
            "home_goals": hg,
            "away_goals": ag,
            "winner": winner,
            "source": source,
        })
    gs_df = pd.DataFrame(gs_results)

    standings = build_standings(gs_df, name_to_group, get_elo)
    thirds = standings[standings["rank"] == 3].sort_values(
        ["pts", "gd", "gf", "elo"], ascending=[False, False, False, False]
    ).reset_index(drop=True)
    thirds["third_rank"] = range(1, len(thirds) + 1)
    thirds["advances"] = thirds["third_rank"] <= 8

    qualifying_thirds = thirds[thirds["advances"]]
    combo_key = "".join(sorted(qualifying_thirds["group"].tolist()))
    alloc_row = alloc_df[alloc_df["qualifying_groups"] == combo_key]
    if alloc_row.empty:
        raise ValueError(f"No third-place allocation row for combination: {combo_key}")
    alloc_row = alloc_row.iloc[0]

    third_slot: dict[int, str] = {}
    for col in ["match_74", "match_77", "match_79", "match_80", "match_81", "match_82", "match_85", "match_87"]:
        match_num = int(col.split("_")[1])
        third_slot[match_num] = alloc_row[col].replace("3", "")
    group_to_third_team = dict(zip(qualifying_thirds["group"], qualifying_thirds["team"]))

    def get_group_finisher(rank: int, group: str) -> str:
        result = standings[(standings["group"] == group) & (standings["rank"] == rank)]
        if result.empty:
            raise ValueError(f"No rank-{rank} team in group {group}")
        return result.iloc[0]["team"]

    def resolve_team(token: str, match_winners: dict[int, str], match_losers: dict[int, str]) -> str:
        m = re.match(r"^W(\d+)$", token)
        if m:
            return match_winners[int(m.group(1))]
        m = re.match(r"^RU(\d+)$", token)
        if m:
            return match_losers[int(m.group(1))]
        m = re.match(r"^([12])([A-L])$", token)
        if m:
            return get_group_finisher(int(m.group(1)), m.group(2))
        raise ValueError(f"Cannot resolve token: {token}")

    def parse_label_tokens(label: str) -> tuple[str, str]:
        parts = label.strip().split(" vs ")
        if len(parts) != 2:
            raise ValueError(f"Unexpected label format: {label}")
        return parts[0].strip(), parts[1].strip()

    stage_names = {2: "Round of 32", 3: "Round of 16", 4: "Quarterfinals",
                   5: "Semifinals", 6: "Bronze Final", 7: "Final"}

    match_winners: dict[int, str] = {}
    match_losers: dict[int, str] = {}
    ko_results = []

    for _, row in matches_df[matches_df["stage_id"] >= 2].sort_values("match_number").iterrows():
        mn = int(row["match_number"])
        stage = int(row["stage_id"])
        tok_a, tok_b = parse_label_tokens(row["match_label"])

        def resolve_with_third(tok):
            if re.match(r"^3[A-L]+$", tok):
                return group_to_third_team[third_slot[mn]]
            return resolve_team(tok, match_winners, match_losers)

        team_a = resolve_with_third(tok_a)
        team_b = resolve_with_third(tok_b)

        actual = completed.get(frozenset({team_a, team_b}))
        if actuals_through_stage_id is not None and stage > actuals_through_stage_id:
            actual = None
        if actual:
            winner = actual["winner"]
            if winner is None:
                hg, ag = actual["home_score"], actual["away_score"]
                if actual["home"] == team_a:
                    winner = team_a if hg > ag else team_b
                else:
                    winner = team_b if hg > ag else team_a
            loser = team_b if winner == team_a else team_a
            hg, ag = actual["home_score"], actual["away_score"]
            if actual["home"] != team_a:
                hg, ag = ag, hg
            source = "actual"
        else:
            winner, loser, _, _ = predict_knockout(team_a, team_b)
            hg, ag, _ = predict_group(team_a, team_b)
            source = "predicted"

        match_winners[mn] = winner
        match_losers[mn] = loser
        ko_results.append({
            "match": mn,
            "match_id": int(row["id"]),
            "stage_id": stage,
            "stage": stage_names[stage],
            "team_a": team_a,
            "team_b": team_b,
            "home_goals": hg,
            "away_goals": ag,
            "winner": winner,
            "loser": loser,
            "source": source,
        })

    return gs_df, pd.DataFrame(ko_results)


def finished_matches_by_stage(
    results_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    id_to_name: dict[int, str],
) -> dict[int, pd.DataFrame]:
    """Return finished matches grouped by stage_id."""
    merged = results_df.merge(matches_df, left_on="match_id", right_on="id")
    merged["home"] = merged["home_team_id"].map(id_to_name)
    merged["away"] = merged["away_team_id"].map(id_to_name)
    by_stage: dict[int, pd.DataFrame] = {}
    for stage_id, grp in merged.groupby("stage_id"):
        by_stage[int(stage_id)] = grp.reset_index(drop=True)
    return by_stage


def evaluate_group_stage(
    model_key: str,
    finished: pd.DataFrame,
    predict_group: PredictGroupFn,
    track_scores: bool = False,
) -> StageAccuracy:
    """Score group-stage win/draw/loss (and optional scoreline) accuracy."""
    correct = 0
    score_correct = 0
    n = len(finished)

    for _, row in finished.iterrows():
        home, away = row["home"], row["away"]
        pred_hg, pred_ag, _ = predict_group(home, away)
        pred_outcome = outcome_from_scores(pred_hg, pred_ag)
        actual_outcome = row["result"]
        if pred_outcome == actual_outcome:
            correct += 1
        if track_scores and pred_hg == int(row["home_score"]) and pred_ag == int(row["away_score"]):
            score_correct += 1

    return StageAccuracy(
        stage_id=1,
        stage_name=STAGE_ID_TO_NAME[1],
        model_key=model_key,
        model_label=MODELS[model_key],
        n_matches=n,
        n_correct=correct,
        win_accuracy=(correct / n * 100) if n else float("nan"),
        n_score_matches=n if track_scores else 0,
        n_score_correct=score_correct if track_scores else 0,
        score_accuracy=(score_correct / n * 100) if track_scores and n else None,
    )


def evaluate_knockout_stage(
    model_key: str,
    stage_id: int,
    ko_df: pd.DataFrame,
    finished: pd.DataFrame,
    track_scores: bool = False,
) -> StageAccuracy | None:
    """
    Score knockout accuracy for one stage.

    Uses predicted rows from ko_df (source='predicted') and compares winners
    to actual results. Only finished matches are scored.
    """
    stage_name = STAGE_ID_TO_NAME[stage_id]
    preds = ko_df[(ko_df["stage_id"] == stage_id) & (ko_df["source"] == "predicted")]
    if preds.empty:
        return None

    # Build lookup of actual winners by team pairing
    actual_by_pair: dict[frozenset, dict] = {}
    for _, row in finished.iterrows():
        actual_by_pair[frozenset({row["home"], row["away"]})] = row

    correct = 0
    score_correct = 0
    n = 0

    for _, pred in preds.iterrows():
        pair = frozenset({pred["team_a"], pred["team_b"]})
        actual = actual_by_pair.get(pair)
        if actual is None:
            continue
        n += 1
        # Actual winner from result code relative to schedule home/away
        if actual["result"] == "H":
            actual_winner = actual["home"]
        else:
            actual_winner = actual["away"]
        if pred["winner"] == actual_winner:
            correct += 1
        if track_scores:
            # Compare predicted scoreline (team_a as "home" in simulation)
            act_hg, act_ag = int(actual["home_score"]), int(actual["away_score"])
            if actual["home"] != pred["team_a"]:
                act_hg, act_ag = act_ag, act_hg
            if int(pred["home_goals"]) == act_hg and int(pred["away_goals"]) == act_ag:
                score_correct += 1

    if n == 0:
        return None

    return StageAccuracy(
        stage_id=stage_id,
        stage_name=stage_name,
        model_key=model_key,
        model_label=MODELS[model_key],
        n_matches=n,
        n_correct=correct,
        win_accuracy=correct / n * 100,
        n_score_matches=n if track_scores else 0,
        n_score_correct=score_correct if track_scores else 0,
        score_accuracy=(score_correct / n * 100) if track_scores else None,
    )


def evaluate_model(
    model_key: str,
    matches_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
    id_to_name: dict[int, str],
    name_to_group: dict[str, str],
    completed: dict[frozenset, dict],
    finished_by_stage: dict[int, pd.DataFrame],
    predict_group: PredictGroupFn,
    predict_knockout: PredictKnockoutFn,
    get_elo: GetEloFn,
) -> AccuracyReport:
    """Compute per-stage accuracy for a single loaded model."""
    report = AccuracyReport(model_key=model_key, model_label=MODELS[model_key])
    track_scores = model_key in SCORELINE_MODELS

    # Group stage — direct predictions (no bracket dependency)
    if 1 in finished_by_stage:
        report.stages.append(
            evaluate_group_stage(
                model_key, finished_by_stage[1], predict_group, track_scores=track_scores
            )
        )

    # Knockout stages — replay bracket with actuals frozen through previous stage
    for stage_id in range(2, 8):
        if stage_id not in finished_by_stage:
            continue
        cutoff = stage_id - 1
        _, ko_df = simulate_tournament(
            matches_df, alloc_df, id_to_name, name_to_group, completed,
            predict_group, predict_knockout, get_elo,
            actuals_through_stage_id=cutoff,
        )
        stage_acc = evaluate_knockout_stage(
            model_key, stage_id, ko_df, finished_by_stage[stage_id],
            track_scores=track_scores,
        )
        if stage_acc is not None:
            report.stages.append(stage_acc)

    return report


def format_accuracy_summary(reports: list[AccuracyReport]) -> str:
    """Pretty-print accuracy tables in the style requested by the user."""
    lines: list[str] = []
    stage_order = sorted({s.stage_id for r in reports for s in r.stages})

    for stage_id in stage_order:
        stage_name = STAGE_ID_TO_NAME[stage_id]
        stage_rows = []
        for report in reports:
            for s in report.stages:
                if s.stage_id == stage_id:
                    stage_rows.append(s)
                    break

        if not stage_rows:
            continue

        lines.append(f"{stage_name} Win Prediction Accuracy:")
        for s in stage_rows:
            pct = f"{s.win_accuracy:.0f}%" if s.n_matches else "n/a"
            lines.append(f"{s.model_label} - {pct}  ({s.n_correct}/{s.n_matches} matches)")
        lines.append("")

        score_rows = [s for s in stage_rows if s.score_accuracy is not None]
        if score_rows:
            lines.append(f"{stage_name} Score Prediction Accuracy:")
            for s in score_rows:
                pct = f"{s.score_accuracy:.0f}%"
                lines.append(f"{s.model_label} - {pct}  ({s.n_score_correct}/{s.n_score_matches} exact scorelines)")
            lines.append("")

    return "\n".join(lines)


def stage_is_complete(matches_df: pd.DataFrame, stage_id: int, now: datetime | None = None) -> bool:
    """True when every match in the stage is past kickoff + result buffer."""
    from scripts.make_stage_snapshot import RESULT_BUFFER

    now = now or datetime.now(timezone.utc)
    stage_matches = matches_df[matches_df["stage_id"] == stage_id]
    if stage_matches.empty:
        return False
    kickoffs = pd.to_datetime(stage_matches["kickoff_at"], utc=True)
    last_finish = kickoffs.max().to_pydatetime() + RESULT_BUFFER
    return now >= last_finish


def ensure_snapshots_for_model(
    model_key: str,
    matches_df: pd.DataFrame,
    force: bool = False,
) -> list[str]:
    """
    Generate any missing per-stage snapshots for one model.

    Returns list of snapshot paths created.
    """
    from scripts.make_stage_snapshot import STAGES, generate_snapshot, snapshot_path

    created: list[str] = []
    now = datetime.now(timezone.utc)
    for stage in STAGES:
        if not stage_is_complete(matches_df, stage.stage_id, now):
            continue
        path = snapshot_path(stage, model_key)
        if path.exists() and not force:
            continue
        generate_snapshot(stage, model_key)
        created.append(str(path))
    return created
