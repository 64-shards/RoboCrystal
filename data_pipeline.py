"""
data_pipeline.py
================
Ingestion & Database Pipeline for the Bipedal Parity Dashboard.

This module:
  1. Pulls historical US manufacturing wage data from the FRED API via pandas_datareader.
  2. Generates a realistic synthetic dataset for bipedal robotics hardware costs.
  3. Merges both datasets by year and pushes the result into a MongoDB Atlas collection.

Usage:
    python data_pipeline.py
"""

import os
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "bipedal_parity")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "historical_costs")

# FRED series: Average Hourly Earnings of Production and
# Nonsupervisory Employees, Manufacturing
FRED_SERIES = "CES3000000008"
START_YEAR = 2010
HOURS_PER_YEAR = 2080  # 40 hrs/week × 52 weeks

# Synthetic robotics cost parameters
ROBOT_COST_2010 = 500_000  # USD in 2010
ANNUAL_DECAY_RATE = 0.10    # 10 % cost reduction per year


# External driver configuration (FRED series IDs)
EXTERNAL_DRIVER_SERIES = {
    "inflation_rate": "FPCPITOTLZGUSA",          # CPI inflation, annual %
    "interest_rate": "FEDFUNDS",                # Effective Federal Funds Rate
    "component_price_proxy": "PPIIDC",          # PPI: Inputs to Industries
    "labor_shortage_proxy": "JTS3000JOL",       # Manufacturing job openings

MONITORING_DIR = Path(__file__).resolve().parent / "monitoring"
SNAPSHOT_DIR = MONITORING_DIR / "snapshots"
RUN_LOG_PATH = MONITORING_DIR / "pipeline_runs.csv"
ALERT_LOG_PATH = MONITORING_DIR / "alert_events.csv"

BASELINE_WINDOW_RUNS = 5
FORECAST_END_YEAR = 2040
MONTE_CARLO_SIMS = 400

ALERT_THRESHOLDS = {
    "hourly_wage_mean_shift": 0.08,
    "annual_salary_mean_shift": 0.08,
    "robot_cost_mean_shift": 0.12,
    "wage_distribution_ks": 0.22,
    "robot_distribution_ks": 0.22,
    "feature_shift": 0.20,
    "parity_year_jump": 2.0,
    "parity_interval_widening": 1.5,
    "parity_probability_swing": 0.15,

}


# ---------------------------------------------------------------------------
# 1. Fetch FRED Manufacturing Wage Data
# ---------------------------------------------------------------------------
def fetch_fred_wages(
    series: str = FRED_SERIES,
    start_year: int = START_YEAR,
    api_key: str = FRED_API_KEY,
) -> pd.DataFrame:
    """
    Pull hourly manufacturing wages from FRED and convert to estimated
    annual salary (hourly_wage × 2080 hours).

    Returns a DataFrame with columns: [year, hourly_wage, annual_salary]
    """
    try:
        import pandas_datareader.data as web

        start = datetime.datetime(start_year, 1, 1)
        end = datetime.datetime.now()

        # pandas_datareader FRED source
        raw = web.DataReader(series, "fred", start, end, api_key=api_key)
        raw = raw.reset_index()
        raw.columns = ["date", "hourly_wage"]

        # Resample to annual average (calendar year)
        raw["year"] = raw["date"].dt.year
        annual = raw.groupby("year")["hourly_wage"].mean().reset_index()
        annual["annual_salary"] = annual["hourly_wage"] * HOURS_PER_YEAR
        annual["annual_salary"] = annual["annual_salary"].round(2)
        annual["hourly_wage"] = annual["hourly_wage"].round(2)

        print(f"[INFO] Fetched {len(annual)} years of FRED wage data.")
        return annual[["year", "hourly_wage", "annual_salary"]]

    except Exception as e:
        print(f"[WARN] FRED fetch failed ({e}). Generating fallback wage data.")
        return _generate_fallback_wages(start_year)


def _generate_fallback_wages(start_year: int) -> pd.DataFrame:
    """
    Fallback: generate realistic wage data if FRED is unavailable.
    Based on BLS historical averages (~$20/hr in 2010, ~3 % annual growth).
    """
    current_year = datetime.datetime.now().year
    years = list(range(start_year, current_year + 1))
    base_wage = 20.00  # approximate 2010 hourly wage
    growth_rate = 0.03  # ~3 % annual growth

    rows = []
    for i, yr in enumerate(years):
        wage = round(base_wage * (1 + growth_rate) ** i, 2)
        rows.append(
            {"year": yr, "hourly_wage": wage, "annual_salary": round(wage * HOURS_PER_YEAR, 2)}
        )
    df = pd.DataFrame(rows)
    print(f"[INFO] Generated fallback wage data for {len(df)} years.")
    return df


# ---------------------------------------------------------------------------
# 1b. External Driver Adapters
# ---------------------------------------------------------------------------
def _fetch_fred_annual_series(
    series: str,
    start_year: int = START_YEAR,
    api_key: str = FRED_API_KEY,
) -> pd.DataFrame | None:
    """Fetch a FRED series and aggregate to annual averages by year."""
    try:
        import pandas_datareader.data as web

        start = datetime.datetime(start_year, 1, 1)
        end = datetime.datetime.now()
        raw = web.DataReader(series, "fred", start, end, api_key=api_key).reset_index()
        raw.columns = ["date", "value"]
        raw["year"] = raw["date"].dt.year
        annual = raw.groupby("year")["value"].mean().reset_index()
        return annual
    except Exception:
        return None


def _generate_fallback_external_drivers(start_year: int = START_YEAR) -> pd.DataFrame:
    """Generate realistic proxy time series for exogenous macro/industry drivers."""
    current_year = datetime.datetime.now().year
    years = np.arange(start_year, current_year + 1)
    t = np.arange(len(years))

    # Smooth synthetic proxies with mild cyclical behaviour + random jitter.
    rng = np.random.default_rng(42)
    inflation = 2.2 + 0.6 * np.sin(t / 2.1) + rng.normal(0, 0.15, len(years))
    interest = 2.0 + 1.3 * np.sin(t / 3.0 + 0.3) + rng.normal(0, 0.2, len(years))
    component = 100 + np.cumsum(1.5 + 0.5 * np.sin(t / 2.6) + rng.normal(0, 0.2, len(years)))
    shortage = 250 + 40 * np.sin(t / 1.8 + 0.8) + rng.normal(0, 8, len(years))

    df = pd.DataFrame(
        {
            "year": years,
            "inflation_rate": np.round(inflation, 2),
            "interest_rate": np.round(np.clip(interest, 0.0, None), 2),
            "component_price_proxy": np.round(component, 2),
            "labor_shortage_proxy": np.round(np.clip(shortage, 0.0, None), 2),
        }
    )
    print(f"[INFO] Generated fallback external drivers for {len(df)} years.")
    return df


def fetch_external_drivers(start_year: int = START_YEAR) -> pd.DataFrame:
    """
    Fetch external forecasting drivers via adapters (FRED + synthetic fallback).

    Returns yearly rows with columns:
      [year, inflation_rate, interest_rate, component_price_proxy, labor_shortage_proxy]
    """
    collected = []
    for col, series in EXTERNAL_DRIVER_SERIES.items():
        annual = _fetch_fred_annual_series(series=series, start_year=start_year)
        if annual is None or annual.empty:
            print(f"[WARN] Could not fetch {col} ({series}) from FRED.")
            return _generate_fallback_external_drivers(start_year)
        annual = annual.rename(columns={"value": col})
        collected.append(annual)

    merged = collected[0]
    for frame in collected[1:]:
        merged = merged.merge(frame, on="year", how="inner")

    # Scale labor shortage proxy down to comparable magnitude.
    merged["labor_shortage_proxy"] = (merged["labor_shortage_proxy"] / 1000.0).round(3)
    merged["component_price_proxy"] = merged["component_price_proxy"].round(2)
    merged["inflation_rate"] = merged["inflation_rate"].round(2)
    merged["interest_rate"] = merged["interest_rate"].round(2)

    print(f"[INFO] Fetched external drivers for {len(merged)} years.")
    return merged


# ---------------------------------------------------------------------------
# 2. Generate Synthetic Robotics Hardware Cost Data
# ---------------------------------------------------------------------------
def generate_robot_costs(
    start_year: int = START_YEAR,
    initial_cost: float = ROBOT_COST_2010,
    decay_rate: float = ANNUAL_DECAY_RATE,
) -> pd.DataFrame:
    """
    Simulate the declining amortized cost of a bipedal humanoid robot
    from *start_year* to the present using an exponential decay model.

    Returns a DataFrame with columns: [year, robot_cost]
    """
    current_year = datetime.datetime.now().year
    years = list(range(start_year, current_year + 1))

    costs = []
    for i, yr in enumerate(years):
        # Add slight noise (±2 %) for realism
        noise = np.random.uniform(-0.02, 0.02)
        cost = initial_cost * ((1 - decay_rate) ** i) * (1 + noise)
        costs.append({"year": yr, "robot_cost": round(cost, 2)})

    df = pd.DataFrame(costs)
    print(f"[INFO] Generated synthetic robot cost data for {len(df)} years.")
    return df


# ---------------------------------------------------------------------------
# 3. Merge & Push to MongoDB Atlas
# ---------------------------------------------------------------------------
def merge_datasets(
    wages_df: pd.DataFrame,
    robot_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge wages, robot costs, and external driver adapters by year."""
    merged = pd.merge(wages_df, robot_df, on="year", how="inner")
    merged = pd.merge(merged, drivers_df, on="year", how="inner")
    print(f"[INFO] Merged dataset has {len(merged)} rows.")
    return merged


def build_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lagged and transformed yearly features for forecasting models.

    Feature examples:
      - YoY deltas (pct and absolute)
      - Rolling means
      - Policy flags
      - Lagged macro drivers
    """
    feat = df.sort_values("year").copy()

    # YoY transforms
    feat["salary_yoy_pct"] = feat["annual_salary"].pct_change()
    feat["robot_cost_yoy_pct"] = feat["robot_cost"].pct_change()
    feat["inflation_delta"] = feat["inflation_rate"].diff()
    feat["interest_delta"] = feat["interest_rate"].diff()
    feat["component_price_yoy_pct"] = feat["component_price_proxy"].pct_change()
    feat["labor_shortage_delta"] = feat["labor_shortage_proxy"].diff()

    # Rolling statistics
    feat["inflation_roll3"] = feat["inflation_rate"].rolling(window=3, min_periods=1).mean()
    feat["interest_roll3"] = feat["interest_rate"].rolling(window=3, min_periods=1).mean()
    feat["component_price_roll3"] = feat["component_price_proxy"].rolling(window=3, min_periods=1).mean()
    feat["labor_shortage_roll3"] = feat["labor_shortage_proxy"].rolling(window=3, min_periods=1).mean()

    # Lagged drivers
    feat["inflation_lag1"] = feat["inflation_rate"].shift(1)
    feat["interest_lag1"] = feat["interest_rate"].shift(1)
    feat["component_price_lag1"] = feat["component_price_proxy"].shift(1)
    feat["labor_shortage_lag1"] = feat["labor_shortage_proxy"].shift(1)

    # Policy / regime flags
    feat["tight_money_policy_flag"] = (feat["interest_rate"] >= 4.0).astype(int)
    feat["high_inflation_flag"] = (feat["inflation_rate"] >= 3.0).astype(int)

    # Fill edge NaNs from lag/diff/pct_change operations.
    feat = feat.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return feat


def push_to_mongodb(df: pd.DataFrame) -> bool:
    """
    Connect to MongoDB Atlas and upsert the merged dataset into the
    configured collection. Returns True on success.
    """
    if not MONGO_URI:
        print("[WARN] MONGO_URI not set. Skipping MongoDB push.")
        print("[INFO] Data will be saved to local CSV fallback instead.")
        _save_local_fallback(df)
        return False

    try:
        from pymongo import MongoClient

        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Verify connection
        client.admin.command("ping")
        print("[INFO] Connected to MongoDB Atlas successfully.")

        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]

        # Clear old data and insert fresh
        collection.delete_many({})
        records = df.to_dict(orient="records")
        collection.insert_many(records)
        print(f"[INFO] Inserted {len(records)} records into '{MONGO_COLLECTION}'.")

        client.close()
        return True

    except Exception as e:
        print(f"[ERROR] MongoDB push failed: {e}")
        print("[INFO] Saving local CSV fallback.")
        _save_local_fallback(df)
        return False


def _save_local_fallback(df: pd.DataFrame) -> None:
    """Save dataset to a local CSV file as a fallback when MongoDB is unavailable."""
    fallback_path = os.path.join(os.path.dirname(__file__), "historical_costs.csv")
    df.to_csv(fallback_path, index=False)
    print(f"[INFO] Fallback data saved to {fallback_path}")


def _ensure_monitoring_dirs() -> None:
    """Create monitoring directories if they do not already exist."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _forecast_curves(df: pd.DataFrame, degree: int = 2):
    """Build labor and robot cost projections through the forecast horizon."""
    years = df["year"].to_numpy(dtype=float)
    labor = df["annual_salary"].to_numpy(dtype=float)
    robot = df["robot_cost"].to_numpy(dtype=float)

    all_years = np.arange(int(years.min()), FORECAST_END_YEAR + 1)
    labor_coeffs = np.polyfit(years, labor, degree)
    robot_exp_coeffs = np.polyfit(years, np.log(robot), 1)

    labor_proj = np.polyval(labor_coeffs, all_years)
    robot_proj = np.exp(np.polyval(robot_exp_coeffs, all_years))
    return all_years, labor_proj, robot_proj


def _find_parity_year(all_years: np.ndarray, labor_proj: np.ndarray, robot_proj: np.ndarray):
    """Find the first year where robot costs drop below labor costs."""
    diff = robot_proj - labor_proj
    for i in range(1, len(diff)):
        if diff[i - 1] > 0 and diff[i] <= 0:
            frac = diff[i - 1] / (diff[i - 1] - diff[i])
            return all_years[i - 1] + frac
    if diff[0] <= 0:
        return float(all_years[0])
    return np.nan


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a two-sample KS statistic without external dependencies."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    data_all = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, data_all, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, data_all, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _compute_input_metrics(df: pd.DataFrame) -> dict:
    """Generate summary input metrics for drift monitoring."""
    ordered = df.sort_values("year")
    wage_growth = ordered["hourly_wage"].pct_change(periods=3).iloc[-1]
    robot_decline = -ordered["robot_cost"].pct_change(periods=3).iloc[-1]

    return {
        "latest_hourly_wage": float(ordered["hourly_wage"].iloc[-1]),
        "latest_annual_salary": float(ordered["annual_salary"].iloc[-1]),
        "latest_robot_cost": float(ordered["robot_cost"].iloc[-1]),
        "wage_mean_5y": float(ordered["hourly_wage"].tail(5).mean()),
        "robot_mean_5y": float(ordered["robot_cost"].tail(5).mean()),
        "wage_growth_3y": float(0.0 if pd.isna(wage_growth) else wage_growth),
        "robot_decline_3y": float(0.0 if pd.isna(robot_decline) else robot_decline),
    }


def _compute_output_metrics(df: pd.DataFrame) -> dict:
    """Generate parity point estimate, interval width, and probability metrics."""
    all_years, labor_proj, robot_proj = _forecast_curves(df)
    parity_point = _find_parity_year(all_years, labor_proj, robot_proj)

    residual_labor = df["annual_salary"].to_numpy(dtype=float) - np.polyval(
        np.polyfit(df["year"], df["annual_salary"], 2), df["year"]
    )
    residual_robot = np.log(df["robot_cost"].to_numpy(dtype=float)) - np.polyval(
        np.polyfit(df["year"], np.log(df["robot_cost"]), 1), df["year"]
    )
    labor_noise = max(float(np.std(residual_labor)), 1.0)
    robot_noise = max(float(np.std(residual_robot)), 1e-4)

    parity_samples = []
    for _ in range(MONTE_CARLO_SIMS):
        sim_labor = labor_proj + np.random.normal(0, labor_noise, len(labor_proj))
        sim_robot = robot_proj * np.exp(np.random.normal(0, robot_noise, len(robot_proj)))
        sample_year = _find_parity_year(all_years, sim_labor, sim_robot)
        parity_samples.append(sample_year)

    samples = np.array(parity_samples, dtype=float)
    valid = samples[~np.isnan(samples)]
    probability = float(len(valid) / len(samples))

    if len(valid):
        lower = float(np.percentile(valid, 10))
        upper = float(np.percentile(valid, 90))
        interval_width = upper - lower
    else:
        lower = np.nan
        upper = np.nan
        interval_width = np.nan

    return {
        "parity_year": float(parity_point) if not np.isnan(parity_point) else np.nan,
        "parity_probability": probability,
        "parity_interval_low": lower,
        "parity_interval_high": upper,
        "parity_interval_width": float(interval_width) if not np.isnan(interval_width) else np.nan,
    }


def _severity(value: float, threshold: float) -> str:
    ratio = value / threshold if threshold else 0
    if ratio >= 1.6:
        return "critical"
    if ratio >= 1.0:
        return "warning"
    return "info"


def _generate_alerts(
    run_id: str,
    current_df: pd.DataFrame,
    current_metrics: dict,
    baseline_runs: pd.DataFrame,
) -> list[dict]:
    """Create alert events by comparing latest run metrics against baseline windows."""
    if baseline_runs.empty:
        return []

    alerts = []
    baseline_mean = baseline_runs.mean(numeric_only=True)

    mean_shift_checks = [
        ("input_drift", "hourly_wage_mean_shift", "latest_hourly_wage"),
        ("input_drift", "annual_salary_mean_shift", "latest_annual_salary"),
        ("input_drift", "robot_cost_mean_shift", "latest_robot_cost"),
    ]
    for category, threshold_key, metric_key in mean_shift_checks:
        base = float(baseline_mean.get(metric_key, np.nan))
        if not np.isfinite(base) or base == 0:
            continue
        current = float(current_metrics[metric_key])
        magnitude = abs((current - base) / base)
        threshold = ALERT_THRESHOLDS[threshold_key]
        if magnitude >= threshold:
            alerts.append(
                {
                    "run_id": run_id,
                    "category": category,
                    "metric": metric_key,
                    "severity": _severity(magnitude, threshold),
                    "value": round(current, 6),
                    "baseline": round(base, 6),
                    "delta": round(current - base, 6),
                    "message": f"{metric_key} shifted by {magnitude:.1%} vs baseline window.",
                }
            )

    baseline_dfs = []
    for rid in baseline_runs["run_id"].tolist():
        path = SNAPSHOT_DIR / f"{rid}.csv"
        if path.exists():
            baseline_dfs.append(pd.read_csv(path))
    if baseline_dfs:
        baseline_df = pd.concat(baseline_dfs, ignore_index=True)
        for field, threshold_key in [("hourly_wage", "wage_distribution_ks"), ("robot_cost", "robot_distribution_ks")]:
            ks = _ks_statistic(current_df[field].to_numpy(dtype=float), baseline_df[field].to_numpy(dtype=float))
            threshold = ALERT_THRESHOLDS[threshold_key]
            if ks >= threshold:
                alerts.append(
                    {
                        "run_id": run_id,
                        "category": "input_drift",
                        "metric": f"{field}_ks",
                        "severity": _severity(ks, threshold),
                        "value": round(ks, 6),
                        "baseline": threshold,
                        "delta": round(ks - threshold, 6),
                        "message": f"{field} distribution drift KS={ks:.3f} exceeded {threshold:.2f}.",
                    }
                )

    feature_current = float(
        np.linalg.norm([current_metrics["wage_growth_3y"], current_metrics["robot_decline_3y"]])
    )
    feature_baseline = float(
        np.linalg.norm(
            [
                baseline_mean.get("wage_growth_3y", 0.0),
                baseline_mean.get("robot_decline_3y", 0.0),
            ]
        )
    )
    if feature_baseline > 0:
        feature_shift = abs((feature_current - feature_baseline) / feature_baseline)
        threshold = ALERT_THRESHOLDS["feature_shift"]
        if feature_shift >= threshold:
            alerts.append(
                {
                    "run_id": run_id,
                    "category": "input_drift",
                    "metric": "feature_vector_shift",
                    "severity": _severity(feature_shift, threshold),
                    "value": round(feature_current, 6),
                    "baseline": round(feature_baseline, 6),
                    "delta": round(feature_current - feature_baseline, 6),
                    "message": f"Feature trend vector changed by {feature_shift:.1%}.",
                }
            )

    output_checks = [
        ("parity_year", "parity_year_jump"),
        ("parity_interval_width", "parity_interval_widening"),
        ("parity_probability", "parity_probability_swing"),
    ]
    for metric_key, threshold_key in output_checks:
        current = current_metrics.get(metric_key)
        base = baseline_mean.get(metric_key, np.nan)
        if not np.isfinite(base) or not np.isfinite(current):
            continue
        magnitude = abs(float(current) - float(base))
        threshold = ALERT_THRESHOLDS[threshold_key]
        if magnitude >= threshold:
            alerts.append(
                {
                    "run_id": run_id,
                    "category": "output_drift",
                    "metric": metric_key,
                    "severity": _severity(magnitude, threshold),
                    "value": round(float(current), 6),
                    "baseline": round(float(base), 6),
                    "delta": round(float(current) - float(base), 6),
                    "message": f"{metric_key} moved by {magnitude:.3f} vs baseline window.",
                }
            )

    return alerts


def _persist_run_and_alerts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist run metrics and alerts, and return updated logs."""
    _ensure_monitoring_dirs()
    run_id = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    input_metrics = _compute_input_metrics(df)
    output_metrics = _compute_output_metrics(df)

    run_record = {
        "run_id": run_id,
        "run_ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **input_metrics,
        **output_metrics,
    }

    run_log = pd.read_csv(RUN_LOG_PATH) if RUN_LOG_PATH.exists() else pd.DataFrame()
    baseline = run_log.tail(BASELINE_WINDOW_RUNS)
    alerts = _generate_alerts(run_id, df, run_record, baseline)

    run_log = pd.concat([run_log, pd.DataFrame([run_record])], ignore_index=True)
    run_log.to_csv(RUN_LOG_PATH, index=False)
    df.to_csv(SNAPSHOT_DIR / f"{run_id}.csv", index=False)

    if ALERT_LOG_PATH.exists():
        alert_log = pd.read_csv(ALERT_LOG_PATH)
    else:
        alert_log = pd.DataFrame(
            columns=[
                "run_id",
                "event_ts",
                "category",
                "metric",
                "severity",
                "value",
                "baseline",
                "delta",
                "message",
                "is_active",
            ]
        )

    if not alert_log.empty:
        alert_log["is_active"] = False

    if alerts:
        events = pd.DataFrame(alerts)
        events["event_ts"] = run_record["run_ts"]
        events["is_active"] = True
        alert_log = pd.concat([alert_log, events], ignore_index=True)

    alert_log.to_csv(ALERT_LOG_PATH, index=False)
    return run_log, alert_log


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def run_pipeline() -> pd.DataFrame:
    """Execute the full ingestion pipeline and return the merged DataFrame."""
    print("=" * 60)
    print("  Bipedal Parity — Data Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Fetch wages
    wages_df = fetch_fred_wages()

    # Step 2: Generate robot costs
    robot_df = generate_robot_costs()

    # Step 2b: Pull external macro/industry drivers
    drivers_df = fetch_external_drivers()

    # Step 3: Merge
    merged_df = merge_datasets(wages_df, robot_df, drivers_df)

    # Step 3b: Build forecasting features and persist alongside raw values
    merged_df = build_forecast_features(merged_df)

    # Step 4: Push to MongoDB (with local fallback)
    push_to_mongodb(merged_df)

    # Step 5: Persist monitoring snapshots, run metrics, and drift alerts
    run_log, alert_log = _persist_run_and_alerts(merged_df)
    active_count = int(alert_log["is_active"].sum()) if not alert_log.empty else 0
    print(f"[INFO] Monitoring run log updated: {len(run_log)} total runs.")
    print(f"[INFO] Active alerts from latest run: {active_count}.")

    print("\n[DONE] Pipeline complete.")
    return merged_df


if __name__ == "__main__":
    result = run_pipeline()
    print("\nPreview:")
    print(result.to_string(index=False))
