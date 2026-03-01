"""
app.py
======
Bipedal Parity Predictive Dashboard — Streamlit Application.

This module:
  1. Retrieves historical cost data from MongoDB Atlas (or local CSV fallback).
  2. Fits regression models to project labor and robot cost curves to 2040.
  3. Calculates the "Bipedal Parity Year" — the intersection point.
  4. Visualises historical + forecasted data with Plotly.
  5. Calls Google Gemini to generate a punchy executive alert.

Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv


SCENARIOS = {
    "Base": {
        "description": "Continuity case using historical trend rates.",
        "labor_growth_shift": 0.0,
        "robot_decay_shift": 0.0,
        "labor_volatility": 0.007,
        "robot_volatility": 0.02,
    },
    "Aggressive Automation": {
        "description": "Faster robot cost declines and learning curves.",
        "labor_growth_shift": 0.0,
        "robot_decay_shift": 0.03,
        "labor_volatility": 0.008,
        "robot_volatility": 0.03,
    },
    "Labor Tightening": {
        "description": "Wage inflation accelerates due to labor scarcity.",
        "labor_growth_shift": 0.02,
        "robot_decay_shift": 0.0,
        "labor_volatility": 0.012,
        "robot_volatility": 0.02,
    },
    "High Friction Deployment": {
        "description": "Regulatory + integration frictions slow automation savings.",
        "labor_growth_shift": 0.005,
        "robot_decay_shift": -0.015,
        "labor_volatility": 0.01,
        "robot_volatility": 0.03,
    },
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "bipedal_parity")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "historical_costs")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

FORECAST_END_YEAR = 2040

# ---------------------------------------------------------------------------
# 1. Data Retrieval
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    """
    Load the historical cost dataset from MongoDB Atlas.
    Falls back to a local CSV file if the connection fails.
    """
    # Try MongoDB first
    if MONGO_URI:
        try:
            from pymongo import MongoClient

            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            db = client[MONGO_DB]
            collection = db[MONGO_COLLECTION]
            records = list(collection.find({}, {"_id": 0}))
            client.close()

            if records:
                df = pd.DataFrame(records)
                st.sidebar.success("✅ Data loaded from MongoDB Atlas")
                return df
        except Exception as e:
            st.sidebar.warning(f"MongoDB unavailable: {e}")

    # Fallback: local CSV
    csv_path = os.path.join(os.path.dirname(__file__), "historical_costs.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.sidebar.info("📂 Data loaded from local CSV fallback")
        return df

    # Last resort: run the pipeline on the fly
    st.sidebar.warning("⚠️ No data source found — running pipeline…")
    from data_pipeline import run_pipeline
    return run_pipeline()


# ---------------------------------------------------------------------------
# 2. Forecasting Engine
# ---------------------------------------------------------------------------

def forecast_costs(df: pd.DataFrame, degree: int = 2):
    """
    Fit polynomial regression models to both the labor cost and robot cost
    curves, then project forward to FORECAST_END_YEAR.

    Returns:
        future_years  — array of years from last historical year+1 to 2040
        labor_proj    — projected annual salary values
        robot_proj    — projected robot cost values
        labor_coeffs  — polynomial coefficients for labor curve
        robot_coeffs  — polynomial coefficients for robot curve
    """
    years = df["year"].values
    labor = df["annual_salary"].values
    robot = df["robot_cost"].values

    # Fit polynomial models
    labor_coeffs = np.polyfit(years, labor, degree)
    robot_coeffs = np.polyfit(years, robot, degree)

    # For the robot curve, also fit an exponential decay for better extrapolation
    # log(cost) = a*year + b  →  cost = exp(a*year + b)
    log_robot = np.log(robot)
    robot_exp_coeffs = np.polyfit(years, log_robot, 1)  # linear in log-space

    # Build future year range (include historical for smooth plotting)
    all_years = np.arange(years.min(), FORECAST_END_YEAR + 1)
    future_years = np.arange(years.max() + 1, FORECAST_END_YEAR + 1)

    # Project labor using polynomial (captures upward trend)
    labor_all = np.polyval(labor_coeffs, all_years)

    # Project robot using exponential decay (more realistic for tech costs)
    robot_all = np.exp(np.polyval(robot_exp_coeffs, all_years))

    return all_years, labor_all, robot_all, labor_coeffs, robot_exp_coeffs


# ---------------------------------------------------------------------------
# 3. Parity Calculation
# ---------------------------------------------------------------------------

def find_parity_year(
    all_years: np.ndarray,
    labor_proj: np.ndarray,
    robot_proj: np.ndarray,
) -> tuple:
    """
    Find the year where robot cost drops below labor cost (parity).

    Returns (parity_year, parity_cost) or (None, None) if no intersection
    within the forecast window.
    """
    diff = robot_proj - labor_proj

    # Look for sign change (robot goes from above to below labor)
    for i in range(1, len(diff)):
        if diff[i - 1] > 0 and diff[i] <= 0:
            # Linear interpolation for sub-year precision
            frac = diff[i - 1] / (diff[i - 1] - diff[i])
            parity_year = all_years[i - 1] + frac
            parity_cost = labor_proj[i - 1] + frac * (labor_proj[i] - labor_proj[i - 1])
            return round(parity_year), round(parity_cost, 2)

    # If robot is already below labor at the start
    if diff[0] <= 0:
        return int(all_years[0]), round(labor_proj[0], 2)

    return None, None


def estimate_baseline_rates(df: pd.DataFrame) -> tuple:
    """Estimate annualized labor growth and robot decay rates from history."""
    years_span = max(df["year"].max() - df["year"].min(), 1)
    labor_rate = (df["annual_salary"].iloc[-1] / df["annual_salary"].iloc[0]) ** (1 / years_span) - 1
    robot_rate = 1 - (df["robot_cost"].iloc[-1] / df["robot_cost"].iloc[0]) ** (1 / years_span)
    return labor_rate, robot_rate


def simulate_parity_distribution(
    df: pd.DataFrame,
    assumptions: dict,
    n_sims: int,
    target_years: list,
    seed: int,
) -> dict:
    """Monte Carlo parity simulation for a single scenario."""
    rng = np.random.default_rng(seed)
    latest = df.sort_values("year").iloc[-1]
    start_year = int(latest["year"])
    labor_start = float(latest["annual_salary"])
    robot_start = float(latest["robot_cost"])

    base_labor_growth, base_robot_decay = estimate_baseline_rates(df)
    labor_growth = max(base_labor_growth + assumptions["labor_growth_shift"], -0.05)
    robot_decay = max(base_robot_decay + assumptions["robot_decay_shift"], -0.1)

    parity_years = []
    for _ in range(n_sims):
        labor_value = labor_start
        robot_value = robot_start
        parity_year = None
        for year in range(start_year + 1, FORECAST_END_YEAR + 1):
            labor_noise = rng.normal(0, assumptions["labor_volatility"])
            robot_noise = rng.normal(0, assumptions["robot_volatility"])

            labor_value *= 1 + labor_growth + labor_noise
            robot_value *= max(1 - robot_decay + robot_noise, 0.6)

            if robot_value <= labor_value:
                parity_year = year
                break
        parity_years.append(parity_year)

    valid_years = [year for year in parity_years if year is not None]
    median_year = int(np.median(valid_years)) if valid_years else None

    probability_by_target = {}
    for year in target_years:
        probability_by_target[year] = round(
            100 * np.mean([(p is not None) and (p <= year) for p in parity_years]), 1
        )

    return {
        "parity_years": parity_years,
        "median_parity_year": median_year,
        "probability_by_target": probability_by_target,
        "effective_rates": {
            "labor_growth": labor_growth,
            "robot_decay": robot_decay,
        },
    }


def build_parity_distribution_overlay(results: dict) -> go.Figure:
    """Overlay histograms of parity year distributions across scenarios."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (scenario_name, result) in enumerate(results.items()):
        valid = [p for p in result["parity_years"] if p is not None]
        if not valid:
            continue
        fig.add_trace(
            go.Histogram(
                x=valid,
                histnorm="probability density",
                name=scenario_name,
                opacity=0.5,
                marker_color=colors[i % len(colors)],
                xbins=dict(start=min(valid) - 0.5, end=FORECAST_END_YEAR + 0.5, size=1),
            )
        )

    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        height=460,
        title="Parity Distribution Overlay by Scenario",
        xaxis_title="Parity Year",
        yaxis_title="Density",
    )
    return fig


def build_decision_brief(
    selected_scenarios: list,
    scenario_results: dict,
    target_years: list,
    latest_year: int,
) -> str:
    """Build a markdown decision brief for download."""
    lines = [
        "# Bipedal Parity Decision Brief",
        "",
        "## Saved assumptions by scenario",
    ]
    for name in selected_scenarios:
        assumptions = SCENARIOS[name]
        lines.extend(
            [
                f"### {name}",
                f"- Context: {assumptions['description']}",
                f"- Labor growth shift: {assumptions['labor_growth_shift']:+.2%}",
                f"- Robot decay shift: {assumptions['robot_decay_shift']:+.2%}",
                f"- Labor volatility: ±{assumptions['labor_volatility']:.2%}",
                f"- Robot volatility: ±{assumptions['robot_volatility']:.2%}",
                "",
            ]
        )

    lines.append("## Key parity probabilities")
    for name in selected_scenarios:
        result = scenario_results[name]
        lines.append(f"### {name}")
        lines.append(
            f"- Median parity year: {result['median_parity_year'] if result['median_parity_year'] else f'>{FORECAST_END_YEAR}'}"
        )
        for year in target_years:
            lines.append(f"- P(parity by {year}): {result['probability_by_target'][year]:.1f}%")
        lines.append("")

    base_result = scenario_results.get("Base")
    if base_result:
        base_median = base_result["median_parity_year"]
        lines.append("## Recommended actions")
        for name in selected_scenarios:
            if name == "Base":
                continue
            median_year = scenario_results[name]["median_parity_year"]
            if base_median is None or median_year is None:
                action = "Maintain optionality: no robust parity signal before forecast horizon."
            elif median_year < base_median:
                delta = base_median - median_year
                action = (
                    f"Accelerate automation pilots now; scenario gains ~{delta} year(s) vs base. "
                    "Prioritize integrator partnerships and capex readiness."
                )
            else:
                delta = median_year - base_median
                action = (
                    f"Stage investments with decision gates; scenario loses ~{delta} year(s) vs base. "
                    "Focus on labor productivity and phased retrofits."
                )
            lines.append(f"- **{name}:** {action}")

    lines.extend(
        [
            "",
            "## Notes",
            f"- Baseline reference year: {latest_year}",
            f"- Forecast horizon: through {FORECAST_END_YEAR}",
            "- This brief is generated from stochastic simulations and should be paired with operating constraints and financing assumptions.",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Plotly Visualisation
# ---------------------------------------------------------------------------

def build_chart(
    df: pd.DataFrame,
    all_years: np.ndarray,
    labor_proj: np.ndarray,
    robot_proj: np.ndarray,
    parity_year,
    parity_cost,
) -> go.Figure:
    """Create an interactive Plotly chart with historical data, forecasts, and parity marker."""

    fig = go.Figure()

    # Historical labor cost
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["annual_salary"],
            mode="markers+lines",
            name="Labor Cost (Historical)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=7),
        )
    )

    # Historical robot cost
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["robot_cost"],
            mode="markers+lines",
            name="Robot Cost (Historical)",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=7),
        )
    )

    # Forecasted labor cost
    fig.add_trace(
        go.Scatter(
            x=all_years,
            y=labor_proj,
            mode="lines",
            name="Labor Cost (Forecast)",
            line=dict(color="#1f77b4", width=2, dash="dash"),
        )
    )

    # Forecasted robot cost
    fig.add_trace(
        go.Scatter(
            x=all_years,
            y=robot_proj,
            mode="lines",
            name="Robot Cost (Forecast)",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )

    # Parity intersection point
    if parity_year is not None:
        fig.add_trace(
            go.Scatter(
                x=[parity_year],
                y=[parity_cost],
                mode="markers+text",
                name=f"Parity ({parity_year})",
                marker=dict(color="red", size=16, symbol="star"),
                text=[f"  PARITY {parity_year}"],
                textposition="top right",
                textfont=dict(size=14, color="red"),
            )
        )

        # Vertical reference line at parity
        fig.add_vline(
            x=parity_year,
            line_dash="dot",
            line_color="red",
            opacity=0.5,
            annotation_text=f"Parity Year: {parity_year}",
            annotation_position="top",
        )

    fig.update_layout(
        title="Bipedal Parity Forecast: Labor vs. Humanoid Robot Cost",
        xaxis_title="Year",
        yaxis_title="Annual Cost (USD)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
    )

    return fig


# ---------------------------------------------------------------------------
# 5. LLM Signal — Google Gemini
# ---------------------------------------------------------------------------

def generate_executive_alert(
    parity_year, current_labor: float, current_robot: float
) -> str:
    """
    Call Google Gemini to produce a 3-sentence executive alert based on the
    parity analysis.
    """
    if not GEMINI_API_KEY:
        return (
            "_Gemini API key not configured._ Set `GEMINI_API_KEY` in your `.env` "
            "file to enable AI-generated executive alerts."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = (
            "You are a logistics and venture capital AI. "
            f"The data shows humanoid parity will be reached in {parity_year}. "
            f"The current annual labor cost for a US manufacturing worker is ${current_labor:,.0f}, "
            f"while the current amortized cost of a bipedal humanoid robot is ${current_robot:,.0f}. "
            "Write a punchy, 3-sentence executive alert advising an engineering firm "
            "on how to adjust their supply chain roadmap."
        )

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"_Gemini API error:_ {e}"


# ---------------------------------------------------------------------------
# 6. Streamlit UI Layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Bipedal Parity Dashboard",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Bipedal Parity Predictive Dashboard")
    st.caption(
        "Forecasting the year when humanoid robot costs drop below "
        "human manufacturing worker costs."
    )

    # --- Load Data ---
    with st.spinner("Loading historical data…"):
        df = load_data()

    if df is None or df.empty:
        st.error("No data available. Please run `data_pipeline.py` first.")
        st.stop()

    # --- Forecasting ---
    with st.spinner("Running forecast models…"):
        all_years, labor_proj, robot_proj, _, _ = forecast_costs(df)
        parity_year, parity_cost = find_parity_year(all_years, labor_proj, robot_proj)

    # Current values (most recent year in dataset)
    latest = df.sort_values("year").iloc[-1]
    current_labor = latest["annual_salary"]
    current_robot = latest["robot_cost"]

    # ==================== SCENARIO CONTROLS ====================
    st.sidebar.markdown("### 🎯 Scenario Studio")
    selected_scenarios = st.sidebar.multiselect(
        "Select scenarios to run in one execution",
        list(SCENARIOS.keys()),
        default=["Base", "Aggressive Automation", "Labor Tightening"],
    )
    simulation_runs = st.sidebar.slider("Monte Carlo runs", min_value=200, max_value=5000, value=1200, step=100)
    target_years = st.sidebar.multiselect(
        "Target years for parity probability",
        options=list(range(int(latest["year"]) + 1, FORECAST_END_YEAR + 1)),
        default=[2030, 2035, 2040],
    )
    if not selected_scenarios:
        st.error("Please select at least one scenario.")
        st.stop()
    if not target_years:
        st.error("Please choose at least one target year for comparison.")
        st.stop()

    scenario_results = {}
    for index, scenario_name in enumerate(selected_scenarios):
        scenario_results[scenario_name] = simulate_parity_distribution(
            df=df,
            assumptions=SCENARIOS[scenario_name],
            n_sims=simulation_runs,
            target_years=target_years,
            seed=42 + index,
        )

    # ==================== TOP METRIC ====================
    st.markdown("---")
    base_result = scenario_results.get("Base")
    base_display_year = base_result["median_parity_year"] if base_result else parity_year
    base_display_cost = parity_cost
    if base_display_year:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #ff4b4b 0%, #ff8c00 100%);
                        padding: 30px; border-radius: 12px; text-align: center;
                        margin-bottom: 20px;">
                <h1 style="color: white; font-size: 2.8em; margin: 0;">
                    🚨 ALERT: Base Scenario Parity Forecasted for {base_display_year}
                </h1>
                <p style="color: #ffffffcc; font-size: 1.2em; margin-top: 10px;">
                    Deterministic crossover estimate: ${base_display_cost:,.0f} / year
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "⚠️ Parity not reached within the forecast window (through 2040)."
        )

    # ==================== KPI COLUMNS ====================
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Labor Cost", f"${current_labor:,.0f}/yr")
    col2.metric("Current Robot Cost", f"${current_robot:,.0f}/yr")
    if base_display_year:
        years_away = base_display_year - int(latest["year"])
        col3.metric("Years to Base Parity", f"{years_away} yrs", delta=f"Target: {base_display_year}")

    # ==================== EXECUTIVE ALERT (GEMINI) ====================
    st.markdown("---")
    st.subheader("📋 Executive Alert — AI Strategic Signal")

    if base_display_year:
        with st.spinner("Generating AI executive alert via Gemini…"):
            alert_text = generate_executive_alert(base_display_year, current_labor, current_robot)
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;
                        border-left: 5px solid #ff4b4b; margin-bottom: 20px;">
                <p style="font-size: 1.1em; line-height: 1.6;">{alert_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Parity not reached — no strategic alert generated.")

    # ==================== PLOTLY CHART ====================
    st.markdown("---")
    st.subheader("📈 Cost Trajectory & Parity Forecast")

    fig = build_chart(df, all_years, labor_proj, robot_proj, parity_year, parity_cost)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🧭 Scenario Forecast Comparison")

    distribution_fig = build_parity_distribution_overlay(scenario_results)
    st.plotly_chart(distribution_fig, use_container_width=True)

    comparison_rows = []
    base_median_year = scenario_results["Base"]["median_parity_year"] if "Base" in scenario_results else None
    for scenario_name in selected_scenarios:
        result = scenario_results[scenario_name]
        row = {
            "Scenario": scenario_name,
            "Median Parity Year": result["median_parity_year"] if result["median_parity_year"] else f">{FORECAST_END_YEAR}",
            "Δ vs Base (years)": "—",
        }
        if base_median_year and result["median_parity_year"]:
            row["Δ vs Base (years)"] = result["median_parity_year"] - base_median_year
        for year in sorted(target_years):
            row[f"P(parity by {year})"] = f"{result['probability_by_target'][year]:.1f}%"
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True)

    st.caption("Negative delta means faster parity than Base (years gained). Positive delta means delay (years lost).")

    st.markdown("---")
    st.subheader("📝 Export Decision Brief")
    brief_text = build_decision_brief(
        selected_scenarios=selected_scenarios,
        scenario_results=scenario_results,
        target_years=sorted(target_years),
        latest_year=int(latest["year"]),
    )
    st.download_button(
        label="Export decision brief (Markdown)",
        data=brief_text,
        file_name="bipedal_parity_decision_brief.md",
        mime="text/markdown",
    )
    with st.expander("Preview brief"):
        st.markdown(brief_text)

    # ==================== RAW DATA ====================
    with st.expander("📊 View Historical Data Table"):
        st.dataframe(
            df.style.format(
                {"annual_salary": "${:,.0f}", "robot_cost": "${:,.0f}", "hourly_wage": "${:.2f}"}
            ),
            use_container_width=True,
        )

    # ==================== SIDEBAR INFO ====================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Configuration")
    st.sidebar.markdown(f"**Forecast horizon:** {FORECAST_END_YEAR}")
    st.sidebar.markdown(f"**MongoDB:** {'Connected' if MONGO_URI else 'Local CSV mode'}")
    st.sidebar.markdown(f"**Gemini API:** {'Configured' if GEMINI_API_KEY else 'Not set'}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built for the **Bipedal Parity Hackathon** 🏗️\n\n"
        "Run `python data_pipeline.py` to refresh data."
    )


if __name__ == "__main__":
    main()
