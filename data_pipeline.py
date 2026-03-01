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

# Segment dimensions + simple multipliers to create segment-level outputs
SEGMENT_VALUES = {
    "region": ["north_america", "europe", "asia"],
    "industry": ["automotive", "electronics", "logistics"],
    "task_type": ["assembly", "inspection", "material_handling"],
}

WAGE_MULTIPLIERS = {
    "region": {"north_america": 1.00, "europe": 0.95, "asia": 0.75},
    "industry": {"automotive": 1.08, "electronics": 1.00, "logistics": 0.92},
    "task_type": {"assembly": 1.00, "inspection": 1.05, "material_handling": 0.90},
}

ROBOT_MULTIPLIERS = {
    "region": {"north_america": 1.05, "europe": 1.02, "asia": 0.92},
    "industry": {"automotive": 0.98, "electronics": 1.00, "logistics": 0.95},
    "task_type": {"assembly": 1.00, "inspection": 1.08, "material_handling": 0.93},
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
def merge_datasets(wages_df: pd.DataFrame, robot_df: pd.DataFrame) -> pd.DataFrame:
    """Merge wage and robot cost DataFrames on year."""
    merged = pd.merge(wages_df, robot_df, on="year", how="inner")
    merged = expand_segmented_dataset(merged)
    print(f"[INFO] Merged dataset has {len(merged)} rows.")
    return merged


def expand_segmented_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand year-level data into segment-level rows.

    Adds segmentation keys: region, industry, task_type, segment_id.
    Applies deterministic multipliers so each segment has distinct cost profiles.
    """
    segmented_rows = []

    for _, row in base_df.iterrows():
        for region in SEGMENT_VALUES["region"]:
            for industry in SEGMENT_VALUES["industry"]:
                for task_type in SEGMENT_VALUES["task_type"]:
                    wage_multiplier = (
                        WAGE_MULTIPLIERS["region"][region]
                        * WAGE_MULTIPLIERS["industry"][industry]
                        * WAGE_MULTIPLIERS["task_type"][task_type]
                    )
                    robot_multiplier = (
                        ROBOT_MULTIPLIERS["region"][region]
                        * ROBOT_MULTIPLIERS["industry"][industry]
                        * ROBOT_MULTIPLIERS["task_type"][task_type]
                    )

                    segmented_rows.append(
                        {
                            "year": int(row["year"]),
                            "region": region,
                            "industry": industry,
                            "task_type": task_type,
                            "segment_id": f"{region}|{industry}|{task_type}",
                            "hourly_wage": round(float(row["hourly_wage"]) * wage_multiplier, 2),
                            "annual_salary": round(float(row["annual_salary"]) * wage_multiplier, 2),
                            "robot_cost": round(float(row["robot_cost"]) * robot_multiplier, 2),
                        }
                    )

    return pd.DataFrame(segmented_rows)


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

    # Step 3: Merge
    merged_df = merge_datasets(wages_df, robot_df)

    # Step 4: Push to MongoDB (with local fallback)
    push_to_mongodb(merged_df)

    print("\n[DONE] Pipeline complete.")
    return merged_df


if __name__ == "__main__":
    result = run_pipeline()
    print("\nPreview:")
    print(result.to_string(index=False))
