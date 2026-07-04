"""
Generate permanent "snapshot" predictions notebooks at the moment each World Cup
2026 stage wraps up.

`predictions/updated_predictions.ipynb` always re-simulates the full bracket
using every actual result the API has, so once a later stage kicks off you can
no longer see "pure" predictions for it — they get overwritten by real scores.
This script freezes that view: it runs the notebook with
`ACTUALS_THROUGH_STAGE_ID` set so that only results through the just-finished
stage are used as actuals, and everything after that is always model-predicted
— even if the real matches have already been played by the time the script
runs. The result is committed as a permanent, never-overwritten file under
`predictions/snapshots/`.

Usage:
    python scripts/make_stage_snapshot.py             # generate any missing, ready snapshots (poisson_model)
    python scripts/make_stage_snapshot.py --list       # show stage readiness without generating
    python scripts/make_stage_snapshot.py --stage 1    # force-(re)generate one stage's snapshot
    python scripts/make_stage_snapshot.py --stage 1 --force            # ...even if it already exists
    python scripts/make_stage_snapshot.py --stage 1 --model dc_model   # ...with a different model

Snapshots land in predictions/snapshots/after_<stage_slug>.ipynb (+ a rendered
.html copy for quick viewing without Jupyter). The default model
(poisson_model) keeps that plain filename; any other --model gets a
`_<model>` suffix, e.g. after_group_stage_dc_model.ipynb, so snapshots for
different models never collide.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nbformat
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHES_CSV = REPO_ROOT / "data" / "tournament" / "matches.csv"
NOTEBOOK = REPO_ROOT / "predictions" / "updated_predictions.ipynb"
SNAPSHOT_DIR = REPO_ROOT / "predictions" / "snapshots"

# How long after a match's scheduled kickoff we assume the real-world result is
# in (regulation + possible ET/penalties + time for the API to mark it
# FINISHED). Generous on purpose — better to snapshot a bit late than early.
RESULT_BUFFER = timedelta(hours=4)

# Must match a filename (without .ipynb) under models/.
MODELS = ["elo", "lr_model", "ensemble_model", "poisson_model", "dc_model"]
DEFAULT_MODEL = "poisson_model"


@dataclass(frozen=True)
class Stage:
    stage_id: int
    slug: str
    completed_label: str
    preview_label: str


# Only stages that have a "next stage" worth previewing. (Bronze Final / Final
# have no further stage to predict, so there's nothing to snapshot after them.)
STAGES: list[Stage] = [
    Stage(1, "group_stage", "Group Stage", "Round of 32"),
    Stage(2, "round_of_32", "Round of 32", "Round of 16"),
    Stage(3, "round_of_16", "Round of 16", "Quarterfinals"),
    Stage(4, "quarterfinals", "Quarterfinals", "Semifinals"),
    Stage(5, "semifinals", "Semifinals", "Final"),
]


def stage_is_complete(matches_df: pd.DataFrame, stage_id: int, now: datetime) -> bool:
    stage_matches = matches_df[matches_df["stage_id"] == stage_id]
    if stage_matches.empty:
        return False
    kickoffs = pd.to_datetime(stage_matches["kickoff_at"], utc=True)
    last_expected_finish = kickoffs.max().to_pydatetime() + RESULT_BUFFER
    return now >= last_expected_finish


def snapshot_path(stage: Stage, model: str = DEFAULT_MODEL) -> Path:
    suffix = "" if model == DEFAULT_MODEL else f"_{model}"
    return SNAPSHOT_DIR / f"after_{stage.slug}{suffix}.ipynb"


def _load_clean_notebook(source: Path) -> "nbformat.NotebookNode":
    """Load `source` with all cell outputs cleared.

    The source notebook can accumulate outputs that don't pass strict
    nbformat validation (e.g. from manual edits). Since we're about to
    re-execute it from scratch anyway, starting from a blanked-outputs copy
    sidesteps that validation entirely instead of failing before execution
    even starts.
    """
    nb = nbformat.read(source, as_version=4)
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def generate_snapshot(stage: Stage, model: str = DEFAULT_MODEL) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = snapshot_path(stage, model)
    base_name = out_path.stem

    print(f"Generating snapshot for '{stage.completed_label}' (model={model}) "
          f"-> predictions/snapshots/{base_name}.ipynb ({stage.preview_label} predictions)")

    nb = _load_clean_notebook(NOTEBOOK)

    # Run the ExecutePreprocessor in-process (rather than shelling out to the
    # nbconvert CLI) so we can pin the kernel's working directory explicitly
    # via `resources`, regardless of where the output file itself lives. The
    # notebook's `Path('..')`-relative data/model loading needs cwd=predictions/
    # to resolve to the repo root, exactly as it does for a normal run.
    env_overrides = {
        "ACTUALS_THROUGH_STAGE_ID": str(stage.stage_id),
        "MODEL_NOTEBOOK": model,
    }
    prev_values = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    try:
        ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOK.parent)}})
    finally:
        for key, prev in prev_values.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev

    nbformat.write(nb, out_path)

    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(out_path)],
        check=True,
    )

    print(f"  done: predictions/snapshots/{base_name}.ipynb (+ {base_name}.html)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage", type=int, choices=[s.stage_id for s in STAGES],
        help="Force-generate a specific stage's snapshot (by stage_id), regardless of whether "
             "the stage is detected as finished yet.",
    )
    parser.add_argument(
        "--model", choices=MODELS, default=DEFAULT_MODEL,
        help=f"Which models/*.ipynb to run (default: {DEFAULT_MODEL}). Only affects --stage / --list; "
             f"the no-argument auto mode always uses {DEFAULT_MODEL}, to keep scheduled runs cheap.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if the snapshot file already exists.")
    parser.add_argument("--list", action="store_true", help="Show stage readiness / snapshot status and exit.")
    args = parser.parse_args()

    matches_df = pd.read_csv(MATCHES_CSV)
    now = datetime.now(timezone.utc)

    if args.list:
        print(f"model = {args.model}")
        for stage in STAGES:
            complete = stage_is_complete(matches_df, stage.stage_id, now)
            exists = snapshot_path(stage, args.model).exists()
            status = "snapshot exists" if exists else ("ready to snapshot" if complete else "stage not finished yet")
            print(f"[{stage.stage_id}] after {stage.completed_label:<13} -> {stage.preview_label:<13} : {status}")
        return 0

    if args.stage:
        stage = next(s for s in STAGES if s.stage_id == args.stage)
        if snapshot_path(stage, args.model).exists() and not args.force:
            print(f"Snapshot already exists for '{stage.completed_label}' (model={args.model}). Use --force to regenerate.")
            return 0
        generate_snapshot(stage, args.model)
        return 0

    # No --stage: auto-detect newly-finished stages. Always uses the default
    # model so scheduled/CI runs stay a single, cheap execution per stage.
    generated = 0
    for stage in STAGES:
        if snapshot_path(stage, DEFAULT_MODEL).exists():
            continue
        if stage_is_complete(matches_df, stage.stage_id, now):
            generate_snapshot(stage, DEFAULT_MODEL)
            generated += 1
    if generated == 0:
        print("No new snapshots to generate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
