import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone
import pytz

st.set_page_config(
    layout="wide",
    page_title="WC 2026 · Prediction Model",
    page_icon="⚽",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Remove default top padding */
.block-container { padding-top: 0rem !important; }

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, #1B3A6B22, #0D111700);
    border: 1px solid #1B3A6B;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 0.5rem;
}

/* Metric label */
[data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #888888 !important;
}

/* Hero internals */
.match-meta { color: #888888; font-size: 13px; margin: 0 0 4px 0; }
.kickoff-time { font-size: 20px; font-weight: 600; margin: 0 0 16px 0; }
.team-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}
.team-home { flex: 2; }
.team-away { flex: 2; text-align: right; }
.team-vs { flex: 1; text-align: center; color: #888888; font-size: 22px; font-weight: 300; }
.team-flag { font-size: 48px; display: block; line-height: 1.2; }
.team-name-label { font-size: 22px; font-weight: 700; display: block; }
.team-elo { color: #888888; font-size: 13px; }
.prob-row { display: flex; justify-content: space-around; margin-top: 4px; }
.prob-item { text-align: center; }
.prob-label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #888888; display: block; }
.prob-value { font-size: 28px; font-weight: 700; display: block; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLAGS = {
    "MEX": "🇲🇽", "RSA": "🇿🇦", "KOR": "🇰🇷", "CZE": "🇨🇿",
    "CAN": "🇨🇦", "BIH": "🇧🇦", "QAT": "🇶🇦", "SUI": "🇨🇭",
    "BRA": "🇧🇷", "MAR": "🇲🇦", "HAI": "🇭🇹", "SCO": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "USA": "🇺🇸", "PAR": "🇵🇾", "AUS": "🇦🇺", "TUR": "🇹🇷",
    "GER": "🇩🇪", "CUR": "🏳️", "CIV": "🇨🇮", "ECU": "🇪🇨",
    "NED": "🇳🇱", "JPN": "🇯🇵", "SWE": "🇸🇪", "TUN": "🇹🇳",
    "BEL": "🇧🇪", "EGY": "🇪🇬", "IRN": "🇮🇷", "NZL": "🇳🇿",
    "ESP": "🇪🇸", "CPV": "🇨🇻", "KSA": "🇸🇦", "URU": "🇺🇾",
    "FRA": "🇫🇷", "SEN": "🇸🇳", "IRQ": "🇮🇶", "NOR": "🇳🇴",
    "ARG": "🇦🇷", "ALG": "🇩🇿", "AUT": "🇦🇹", "JOR": "🇯🇴",
    "POR": "🇵🇹", "COD": "🇨🇩", "UZB": "🇺🇿", "COL": "🇨🇴",
    "ENG": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "CRO": "🇭🇷", "GHA": "🇬🇭", "PAN": "🇵🇦",
}

# teams.csv team_name → elo_ratings_wc2026.csv country
ELO_NAME_MAP = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
}

# teams.csv team_name → test.csv team
TEST_NAME_MAP = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
}

DEFAULT_ELO = 1500
ET_TZ = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict_match(elo_home: float, elo_away: float) -> tuple[float, float, float]:
    expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
    home_win = round(expected_home * 0.72, 3)
    away_win = round((1 - expected_home) * 0.72, 3)
    draw = round(1 - home_win - away_win, 3)
    return home_win, draw, away_win


def fmt_kickoff_et(ts: pd.Timestamp) -> str:
    local = ts.astimezone(ET_TZ)
    hour = local.strftime("%I").lstrip("0") or "12"
    return local.strftime(f"%A, %B {local.day} · {hour}:%M %p ET")


def get_flag(fifa_code: str) -> str:
    return FLAGS.get(str(fifa_code), "🏳️")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_all_data() -> dict:
    base = "data/"

    matches = pd.read_csv(base + "matches.csv")
    teams = pd.read_csv(base + "teams.csv")
    cities = pd.read_csv(base + "host_cities.csv")
    stages = pd.read_csv(base + "tournament_stages.csv")
    test = pd.read_csv(base + "test.csv")

    try:
        results = pd.read_csv(base + "results.csv")
    except FileNotFoundError:
        results = pd.DataFrame(
            columns=["match_id", "home_score", "away_score", "result", "played_at"]
        )

    # ELO: most recent snapshot per country
    elo_df = pd.read_csv(base + "elo_ratings_wc2026.csv")
    elo_df["snapshot_date"] = pd.to_datetime(elo_df["snapshot_date"])
    elo_df = elo_df.sort_values("snapshot_date")
    elo_latest = elo_df.groupby("country").last().reset_index()
    elo_dict: dict[str, float] = dict(zip(elo_latest["country"], elo_latest["rating"]))

    # Normalize is_placeholder to bool
    teams["is_placeholder"] = teams["is_placeholder"].astype(str).str.lower() == "true"

    # Parse kickoff to UTC-aware timestamps
    matches["kickoff_at"] = pd.to_datetime(matches["kickoff_at"], utc=True)

    # Build merged matches dataframe
    teams_home = (
        teams.add_suffix("_home")
        .rename(columns={"id_home": "home_team_id"})
    )
    teams_away = (
        teams.add_suffix("_away")
        .rename(columns={"id_away": "away_team_id"})
    )
    cities_keyed = cities.rename(columns={"id": "city_id"})
    stages_keyed = stages.rename(columns={"id": "stage_id"})

    merged = (
        matches
        .merge(teams_home, on="home_team_id")
        .merge(teams_away, on="away_team_id")
        .merge(cities_keyed, on="city_id")
        .merge(stages_keyed, on="stage_id")
    )

    merged = merged[
        (~merged["is_placeholder_home"]) & (~merged["is_placeholder_away"])
    ].copy()

    return {
        "merged": merged,
        "teams": teams,
        "test": test,
        "results": results,
        "elo_dict": elo_dict,
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data = load_all_data()
merged = data["merged"]
elo_dict = data["elo_dict"]
test_df = data["test"]
results_df = data["results"]


def get_elo(team_name: str) -> float:
    lookup = ELO_NAME_MAP.get(team_name, team_name)
    return float(elo_dict.get(lookup, DEFAULT_ELO))


def get_form_row(team_name: str):
    lookup = TEST_NAME_MAP.get(team_name, team_name)
    rows = test_df[test_df["team"] == lookup]
    return rows.iloc[0] if not rows.empty else None


# ---------------------------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------------------------

st.markdown("# ⚽ WC 2026 · Prediction Model")
st.markdown(
    '<p style="color:#888888;margin-top:-12px;margin-bottom:16px;">'
    "ELO-based predictions · Updated after each match"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SECTION 1 — HERO
# ---------------------------------------------------------------------------

now_utc = datetime.now(timezone.utc)

future = merged[merged["kickoff_at"] >= now_utc].sort_values("kickoff_at")
if future.empty:
    hero = merged.sort_values("kickoff_at", ascending=False).iloc[0]
else:
    hero = future.iloc[0]

home_name: str = hero["team_name_home"]
away_name: str = hero["team_name_away"]
home_code: str = hero["fifa_code_home"]
away_code: str = hero["fifa_code_away"]
home_elo = get_elo(home_name)
away_elo = get_elo(away_name)
home_flag = get_flag(home_code)
away_flag = get_flag(away_code)
home_win, draw, away_win = predict_match(home_elo, away_elo)
kickoff_str = fmt_kickoff_et(hero["kickoff_at"])

st.markdown(f"""
<div class="hero-card">
  <p class="match-meta">{hero["match_label"]} &nbsp;·&nbsp; {hero["city_name"]} &nbsp;·&nbsp; {hero["venue_name"]}</p>
  <p class="kickoff-time">{kickoff_str}</p>
  <div class="team-row">
    <div class="team-home">
      <span class="team-flag">{home_flag}</span>
      <span class="team-name-label">{home_name}</span>
      <span class="team-elo">ELO: {int(home_elo):,}</span>
    </div>
    <div class="team-vs">VS</div>
    <div class="team-away">
      <span class="team-flag">{away_flag}</span>
      <span class="team-name-label">{away_name}</span>
      <span class="team-elo">ELO: {int(away_elo):,}</span>
    </div>
  </div>
  <div class="prob-row">
    <div class="prob-item">
      <span class="prob-label">Home Win</span>
      <span class="prob-value">{home_win:.0%}</span>
    </div>
    <div class="prob-item">
      <span class="prob-label">Draw</span>
      <span class="prob-value">{draw:.0%}</span>
    </div>
    <div class="prob-item">
      <span class="prob-label">Away Win</span>
      <span class="prob-value">{away_win:.0%}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Probability stacked bar chart
fig_bar = go.Figure()
for label, val, color in [
    ("Home Win", home_win, "#1B3A6B"),
    ("Draw", draw, "#3D3D4E"),
    ("Away Win", away_win, "#2D5A2D"),
]:
    fig_bar.add_trace(go.Bar(
        name=label,
        x=[val],
        y=[""],
        orientation="h",
        marker_color=color,
        text=[f"{val:.0%}"],
        textposition="inside",
        insidetextanchor="middle",
        showlegend=False,
    ))
fig_bar.update_layout(
    barmode="stack",
    height=40,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False),
)
st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# SECTION 2 — RECENT FORM + HEAD TO HEAD
# ---------------------------------------------------------------------------

st.markdown("---")
col_home_form, col_h2h, col_away_form = st.columns(3)


def render_form(col, team_name: str):
    with col:
        st.markdown(f"**{team_name} Recent Form**")
        row = get_form_row(team_name)
        if row is not None:
            total = row["wins_last_4y"] + row["losses_last_4y"] + row["draws_last_4y"]
            st.metric(
                "Win Rate (Last 4Y)",
                f"{row['wins_last_4y'] / total:.0%}" if total > 0 else "N/A",
            )
            st.metric(
                "Goals/Game",
                f"{row['goals_scored_last_4y'] / total:.1f}" if total > 0 else "N/A",
            )
            st.metric(
                "Goals Against/Game",
                f"{row['goals_received_last_4y'] / total:.1f}" if total > 0 else "N/A",
            )
        else:
            st.info("Form data not available for this team")
        st.caption("Based on last 4 years of international matches")


render_form(col_home_form, home_name)

with col_h2h:
    st.markdown("**Head to Head**")
    st.info("🔜 Head-to-head records coming in the next update")

render_form(col_away_form, away_name)

# ---------------------------------------------------------------------------
# SECTION 3 — BOTTOM STRIP
# ---------------------------------------------------------------------------

st.markdown("---")
col_today, col_accuracy, col_surprise = st.columns(3)

# --- Today's Matches ---
with col_today:
    st.markdown("**Today's Matches**")
    today_utc = now_utc.date()
    today_matches = merged[merged["kickoff_at"].dt.date == today_utc].copy()
    if today_matches.empty:
        st.caption("No matches scheduled today")
    else:
        today_matches["Home"] = today_matches.apply(
            lambda r: f"{get_flag(r['fifa_code_home'])} {r['team_name_home']}", axis=1
        )
        today_matches["Away"] = today_matches.apply(
            lambda r: f"{get_flag(r['fifa_code_away'])} {r['team_name_away']}", axis=1
        )
        today_matches["Kickoff"] = today_matches["kickoff_at"].apply(
            lambda t: t.astimezone(ET_TZ).strftime("%I:%M %p ET").lstrip("0")
        )
        today_matches["Group"] = today_matches["match_label"]
        st.dataframe(
            today_matches[["Home", "Away", "Kickoff", "Group"]],
            hide_index=True,
            use_container_width=True,
        )

# --- Model Accuracy ---
with col_accuracy:
    st.markdown("**Model Accuracy**")
    if results_df.empty or len(results_df) == 0:
        st.metric("Correct Predictions", "0 / 0")
        st.caption("Tournament begins June 11, 2026")
    else:
        correct = 0
        total = 0
        for _, res_row in results_df.iterrows():
            try:
                mid = int(res_row["match_id"])
            except (ValueError, TypeError):
                continue
            match_rows = merged[merged["id"] == mid]
            if match_rows.empty:
                continue
            m = match_rows.iloc[0]
            hw, dr, aw = predict_match(get_elo(m["team_name_home"]), get_elo(m["team_name_away"]))
            probs = {"H": hw, "D": dr, "A": aw}
            predicted = max(probs, key=probs.get)
            if predicted == res_row["result"]:
                correct += 1
            total += 1
        st.metric("Correct Predictions", f"{correct} / {total}")
        if total > 0:
            st.caption(f"{correct / total:.0%} accuracy")

# --- Biggest Surprise ---
with col_surprise:
    st.markdown("**Biggest Surprise**")
    if results_df.empty or len(results_df) == 0:
        st.info("No matches played yet — check back after June 11")
    else:
        min_prob = 1.0
        surprise_label = ""
        surprise_result_txt = ""
        for _, res_row in results_df.iterrows():
            try:
                mid = int(res_row["match_id"])
            except (ValueError, TypeError):
                continue
            match_rows = merged[merged["id"] == mid]
            if match_rows.empty:
                continue
            m = match_rows.iloc[0]
            hw, dr, aw = predict_match(get_elo(m["team_name_home"]), get_elo(m["team_name_away"]))
            probs = {"H": hw, "D": dr, "A": aw}
            actual_prob = probs.get(res_row["result"], 1.0)
            if actual_prob < min_prob:
                min_prob = actual_prob
                hf = get_flag(m["fifa_code_home"])
                af = get_flag(m["fifa_code_away"])
                surprise_label = f"{hf} {m['team_name_home']} vs {af} {m['team_name_away']}"
                result_map = {
                    "H": f"{m['team_name_home']} win",
                    "A": f"{m['team_name_away']} win",
                    "D": "Draw",
                }
                surprise_result_txt = result_map.get(res_row["result"], res_row["result"])

        if surprise_label:
            st.markdown(f"**{surprise_label}**")
            st.markdown(f"Result: {surprise_result_txt}")
            st.caption(f"The model gave this outcome only {min_prob:.0%} chance")
        else:
            st.info("No matches played yet — check back after June 11")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Built by Karl Estampador · ELO-based predictions updated after each match · "
    "Source: football-data.org, Kaggle WC 2026 datasets"
)
