# World Cup 2026 Predictor

A FIFA World Cup 2026 prediction project. It combines a Kaggle dataset of every
international match since 1872 with live results fetched from
[football-data.org](https://www.football-data.org/) to train several match
models (ELO, Logistic Regression, an LR + LightGBM ensemble, Poisson, and
Dixon-Coles), simulate the full tournament bracket from group stage to champion,
and serve a live Streamlit dashboard that updates after each match.

## Folder structure

```
world-cup-predictor/
├── models/                       # Match-prediction model notebooks
│   ├── elo.ipynb                 # Deterministic ELO model
│   ├── lr_model.ipynb            # Logistic Regression (WC pre-tournament features)
│   ├── ensemble_model.ipynb      # Soft-voting LR + LightGBM ensemble
│   ├── poisson_model.ipynb       # Poisson goals model (all internationals)
│   └── dc_model.ipynb            # Dixon-Coles model (all internationals)
├── predictions/
│   ├── original_predictions.ipynb  # Tournament bracket simulation
│   ├── updated_predictions.ipynb   # Same simulation + auto-updating data pipeline
│   └── snapshots/                  # Permanent per-stage prediction snapshots (see below)
├── scripts/
│   ├── data_pipeline.py          # Reusable/CLI pipeline -> master dataset (CSV + SQLite)
│   ├── fetch_results.py          # Fetch finished WC 2026 scores for the dashboard
│   └── make_stage_snapshot.py    # Freeze per-stage predictions into predictions/snapshots/
├── data/
│   ├── historical/               # Kaggle CSVs (results, shootouts, former_names, goalscorers)
│   ├── tournament/               # WC 2026 schedule + team/ELO data
│   ├── live/                     # API-fetched data (wc2026_results.csv, api_matches.csv)
│   ├── combined/                 # master_matches.csv + master_matches.db (built by pipeline)
│   └── logs/                     # update_log.json (one JSON record per pipeline run)
├── streamlit_app.py              # Live dashboard
├── requirements.txt
├── .env.example                  # Copy to .env and add your API key
└── .github/workflows/            # Scheduled results updater
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) configure your API key for live/historical API data
cp .env.example .env
#   then edit .env and set FOOTBALL_DATA_API_KEY=<your token>

# 3. Build the master dataset (historical only without a key)
python scripts/data_pipeline.py

# 4. Run the prediction notebook end-to-end
jupyter nbconvert --to notebook --execute predictions/updated_predictions.ipynb

# 5. Launch the dashboard
streamlit run streamlit_app.py
```

The project runs correctly with **no** API key (historical data only) and with a
key (historical + API data).

## Setting `FOOTBALL_DATA_API_KEY`

Get a free token at [football-data.org](https://www.football-data.org/client/register).
The key is **never** hardcoded. Provide it in one of two ways:

- Environment variable:
  - macOS/Linux: `export FOOTBALL_DATA_API_KEY=your_key_here`
  - Windows (PowerShell): `$env:FOOTBALL_DATA_API_KEY = "your_key_here"`
- `.env` file at the project root (loaded automatically via `python-dotenv`):

  ```
  FOOTBALL_DATA_API_KEY=your_key_here
  ```

## Models

| Notebook | Model | Trained on |
| --- | --- | --- |
| `models/elo.ipynb` | Deterministic ELO; the higher-rated team wins, scoreline derived from the ELO gap. | ELO snapshots for the 2026 field |
| `models/lr_model.ipynb` | Logistic Regression over seven pre-tournament difference features (rank, squad value, form, ...). | World Cup matches 2002-2022 |
| `models/ensemble_model.ipynb` | Soft-voting ensemble of Logistic Regression + LightGBM on the same features; same public API as the LR model. | World Cup matches 2002-2022 |
| `models/poisson_model.ipynb` | Two-component Poisson goals model with attack/defence strengths. | All competitive internationals (via `load_from_db`) |
| `models/dc_model.ipynb` | Dixon-Coles: Poisson with a low-score correlation correction and time-weighted fitting. | All competitive internationals (via `load_from_db`) |

The Poisson and Dixon-Coles notebooks read their training data from the combined
master dataset built by the pipeline (`scripts.data_pipeline.load_from_db`). The
others read the relevant CSVs under `data/`.

## Running the Streamlit app

```bash
streamlit run streamlit_app.py
```

The dashboard reads the WC 2026 schedule/teams/ELO from `data/tournament/` and
live scores from `data/live/wc2026_results.csv`.

## Scheduling the update pipeline

`scripts/data_pipeline.py` is self-contained and reads the API key from the
environment, so it can be scheduled without modification. Example crontab entry
that rebuilds the master dataset every day at 04:00:

```cron
0 4 * * * cd /path/to/world-cup-predictor && FOOTBALL_DATA_API_KEY=your_key_here /usr/bin/python scripts/data_pipeline.py >> data/logs/cron.log 2>&1
```

During the tournament, `.github/workflows/update_results.yml` separately fetches
finished match scores into `data/live/wc2026_results.csv` on a schedule via
`scripts/fetch_results.py`.

## Per-stage prediction snapshots

`updated_predictions.ipynb` always simulates the *entire* remaining bracket
using every actual result the API has, no matter which stage is live. That
means as soon as a new stage kicks off, "pure" predictions for it get
overwritten by real scores on the next run — there's no way to look back at
what the model predicted for, say, Round of 32 before any of those matches
were played.

`predictions/snapshots/` solves this: each file is a permanent, frozen copy of
the notebook's output captured right when a stage finished, containing
predictions for the *next* stage that are never contaminated by real results
from that next stage — even ones played before the snapshot was generated.

| File | Actual results used through | Shows predictions for |
| --- | --- | --- |
| `after_group_stage.ipynb` | Group Stage | Round of 32 |
| `after_round_of_32.ipynb` | Round of 32 | Round of 16 |
| `after_round_of_16.ipynb` | Round of 16 | Quarterfinals |
| `after_quarterfinals.ipynb` | Quarterfinals | Semifinals |
| `after_semifinals.ipynb` | Semifinals | Final |

Each snapshot is generated by running `updated_predictions.ipynb` with the
`ACTUALS_THROUGH_STAGE_ID` environment variable set (see the "Snapshot
control" cell near the top of the notebook); matches in later stages are
always model-predicted regardless of what has actually happened in the real
tournament by that point.

Generate snapshots manually with:

```bash
python scripts/make_stage_snapshot.py            # generate any missing, ready snapshots (poisson_model)
python scripts/make_stage_snapshot.py --list      # check stage readiness without generating
python scripts/make_stage_snapshot.py --stage 1   # force-(re)generate one stage (1=Group Stage, ... 5=Semifinals)
python scripts/make_stage_snapshot.py --stage 1 --model dc_model   # ...with a different model
```

### Snapshotting a different model

`updated_predictions.ipynb` picks its model via the `MODEL_NOTEBOOK`
environment variable (see the "Model selection" cell near the top — defaults
to `poisson_model` when unset), which controls which `models/*.ipynb` gets
`%run`. `--model` on `make_stage_snapshot.py` sets that automatically, so you
can get the same stage's predictions from any model without hand-editing the
notebook:

```bash
python scripts/make_stage_snapshot.py --stage 1 --model elo
python scripts/make_stage_snapshot.py --stage 1 --model lr_model
python scripts/make_stage_snapshot.py --stage 1 --model ensemble_model
python scripts/make_stage_snapshot.py --stage 1 --model dc_model
```

The default model (`poisson_model`) keeps the plain `after_<stage>.ipynb`
filename; any other `--model` gets a `_<model>` suffix, e.g.
`after_group_stage_dc_model.ipynb`, so snapshots for different models never
overwrite each other. The unattended/CI auto-detect mode (no `--stage`)
always uses `poisson_model`, to keep scheduled runs to one execution per
stage — generating other models is a manual, on-demand action.

`.github/workflows/generate_stage_snapshots.yml` runs this automatically
after each `update_results.yml` run during the tournament window and commits
any newly-ready snapshot.
