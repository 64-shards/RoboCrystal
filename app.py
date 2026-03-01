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

from evaluation import compute_forecast_reliability, save_evaluation_history

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "bipedal_parity")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "historical_costs")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

FORECAST_END_YEAR = 2040
MODEL_VERSION = "poly2_labor-exp1_robot_v1"

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
        reliability_metrics = compute_forecast_reliability(df, degree=2)
        latest_history = save_evaluation_history(reliability_metrics, model_version=MODEL_VERSION).iloc[-1]

    # Current values (most recent year in dataset)
    latest = df.sort_values("year").iloc[-1]
    current_labor = latest["annual_salary"]
    current_robot = latest["robot_cost"]

    # ==================== TOP METRIC ====================
    st.markdown("---")
    if parity_year:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #ff4b4b 0%, #ff8c00 100%);
                        padding: 30px; border-radius: 12px; text-align: center;
                        margin-bottom: 20px;">
                <h1 style="color: white; font-size: 2.8em; margin: 0;">
                    🚨 ALERT: Bipedal Parity Forecasted for {parity_year}
                </h1>
                <p style="color: #ffffffcc; font-size: 1.2em; margin-top: 10px;">
                    Estimated crossover cost: ${parity_cost:,.0f} / year
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
    if parity_year:
        years_away = parity_year - int(latest["year"])
        col3.metric("Years to Parity", f"{years_away} yrs", delta=f"Target: {parity_year}")

    # ==================== FORECAST RELIABILITY ====================
    st.markdown("---")
    st.subheader("🛡️ Forecast Reliability")
    latest_coverage = reliability_metrics.get("interval_coverage_rate")
    target_coverage = reliability_metrics.get("coverage_target", 0.80)
    avg_width = reliability_metrics.get("average_interval_width")
    calibration = reliability_metrics.get("calibration_by_horizon", {})

    rel_col1, rel_col2, rel_col3 = st.columns(3)
    if latest_coverage is None:
        rel_col1.metric("P10–P90 Coverage", "N/A")
        rel_col2.metric("Coverage Target", f"{target_coverage:.0%}")
        rel_col3.metric("Avg Interval Width", "N/A")
        st.caption("Not enough data points for reliability backtesting.")
    else:
        rel_col1.metric(
            "P10–P90 Coverage",
            f"{latest_coverage:.1%}",
            delta=f"vs target {target_coverage:.0%}",
        )
        rel_col2.metric("Coverage Target", f"{target_coverage:.0%}")
        rel_col3.metric("Avg Interval Width", f"${avg_width:,.0f}")
        st.caption(
            "Calibration by horizon: "
            + ", ".join(
                f"{h}y={value:.1%}" if value is not None else f"{h}y=N/A"
                for h, value in calibration.items()
            )
        )
        st.caption(
            f"Latest run: {latest_history['timestamp']} • model: {latest_history['model_version']}"
        )

    # ==================== EXECUTIVE ALERT (GEMINI) ====================
    st.markdown("---")
    st.subheader("📋 Executive Alert — AI Strategic Signal")

    if parity_year:
        with st.spinner("Generating AI executive alert via Gemini…"):
            alert_text = generate_executive_alert(parity_year, current_labor, current_robot)
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
