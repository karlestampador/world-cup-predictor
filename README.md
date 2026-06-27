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
│   └── updated_predictions.ipynb   # Same simulation + auto-updating data pipeline
├── scripts/
│   ├── data_pipeline.py          # Reusable/CLI pipeline -> master dataset (CSV + SQLite)
│   └── fetch_results.py          # Fetch finished WC 2026 scores for the dashboard
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
