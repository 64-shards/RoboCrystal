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

    print("\n[DONE] Pipeline complete.")
    return merged_df


if __name__ == "__main__":
    result = run_pipeline()
    print("\nPreview:")
    print(result.to_string(index=False))
