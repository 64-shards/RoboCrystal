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
from data_pipeline import build_forecast_features

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

FEATURE_COLUMNS = [
    "year",
    "inflation_rate",
    "interest_rate",
    "component_price_proxy",
    "labor_shortage_proxy",
    "inflation_delta",
    "interest_delta",
    "component_price_yoy_pct",
    "labor_shortage_delta",
    "inflation_roll3",
    "interest_roll3",
    "component_price_roll3",
    "labor_shortage_roll3",
    "inflation_lag1",
    "interest_lag1",
    "component_price_lag1",
    "labor_shortage_lag1",
    "tight_money_policy_flag",
    "high_inflation_flag",
]


def _project_external_drivers(df: pd.DataFrame, end_year: int = FORECAST_END_YEAR) -> pd.DataFrame:
    """Project external drivers forward with recent linear trend extrapolation."""
    hist = df.sort_values("year").copy()
    last_year = int(hist["year"].max())
    future_years = np.arange(last_year + 1, end_year + 1)
    if len(future_years) == 0:
        return hist

    driver_cols = [
        "inflation_rate",
        "interest_rate",
        "component_price_proxy",
        "labor_shortage_proxy",
    ]

    future_rows = {"year": future_years}
    for col in driver_cols:
        window = min(5, len(hist))
        recent = hist.tail(window)
        x = recent["year"].values
        y = recent[col].values
        slope, intercept = np.polyfit(x, y, 1)
        projected = slope * future_years + intercept

        if col in {"inflation_rate", "interest_rate", "labor_shortage_proxy"}:
            projected = np.clip(projected, 0, None)

        future_rows[col] = projected

    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([hist, future_df], ignore_index=True)
    return combined


def _fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve linear regression with intercept using least squares."""
    X_design = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return coeffs


def _predict_linear_regression(coeffs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Predict using least-squares coefficients."""
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ coeffs


def forecast_costs(df: pd.DataFrame):
    """
    Train feature-matrix regressors for labor and robot cost and project
    yearly values to FORECAST_END_YEAR using transformed macro features.

    Returns:
        all_years     — historical + forecast year axis
        labor_proj    — projected annual salary values
        robot_proj    — projected robot cost values
        labor_coeffs  — linear regression coefficients for labor model
        robot_coeffs  — linear regression coefficients for robot model
    """
    enriched = build_forecast_features(df)
    extended = _project_external_drivers(enriched, FORECAST_END_YEAR)
    feature_ready = build_forecast_features(extended)

    train_mask = feature_ready["year"] <= enriched["year"].max()
    train_df = feature_ready.loc[train_mask].copy()

    X_train = train_df[FEATURE_COLUMNS].values
    labor_train = train_df["annual_salary"].values
    robot_train = train_df["robot_cost"].values

    labor_coeffs = _fit_linear_regression(X_train, labor_train)
    robot_coeffs = _fit_linear_regression(X_train, robot_train)

    all_years = feature_ready["year"].values
    X_all = feature_ready[FEATURE_COLUMNS].values

    labor_all = _predict_linear_regression(labor_coeffs, X_all)
    robot_all = _predict_linear_regression(robot_coeffs, X_all)
    robot_all = np.clip(robot_all, 0, None)

    return all_years, labor_all, robot_all, labor_coeffs, robot_coeffs


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
