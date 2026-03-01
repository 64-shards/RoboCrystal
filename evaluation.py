"""Forecast evaluation utilities for interval reliability tracking."""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

Z_P10_P90 = 1.2815515655446004  # central 80% interval for Normal errors
TARGET_INTERVAL_COVERAGE = 0.80
DEFAULT_HISTORY_PATH = os.path.join(
    os.path.dirname(__file__), "forecast_evaluation_history.csv"
)


def _fit_and_predict(train_df: pd.DataFrame, year: int, degree: int) -> Dict[str, float]:
    """Fit the same forecasting model family as app.forecast_costs and predict one year."""
    years = train_df["year"].to_numpy()
    labor = train_df["annual_salary"].to_numpy()
    robot = train_df["robot_cost"].to_numpy()

    labor_coeffs = np.polyfit(years, labor, degree)
    labor_pred = float(np.polyval(labor_coeffs, year))

    log_robot = np.log(robot)
    robot_exp_coeffs = np.polyfit(years, log_robot, 1)
    robot_pred = float(np.exp(np.polyval(robot_exp_coeffs, year)))

    labor_train_pred = np.polyval(labor_coeffs, years)
    labor_sigma = float(np.std(labor - labor_train_pred, ddof=1)) if len(years) > 2 else 0.0

    robot_train_pred = np.exp(np.polyval(robot_exp_coeffs, years))
    robot_sigma = float(np.std(robot - robot_train_pred, ddof=1)) if len(years) > 2 else 0.0

    return {
        "labor_pred": labor_pred,
        "robot_pred": robot_pred,
        "labor_sigma": max(labor_sigma, 1e-9),
        "robot_sigma": max(robot_sigma, 1e-9),
    }


def _generate_backtest_rows(
    df: pd.DataFrame,
    horizons: Iterable[int],
    degree: int,
    min_train_points: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    ordered = df.sort_values("year").reset_index(drop=True)
    year_to_row = {int(r.year): r for r in ordered.itertuples(index=False)}
    years = ordered["year"].to_numpy()

    for i in range(min_train_points, len(years)):
        train_df = ordered.iloc[:i]
        train_end_year = int(train_df["year"].iloc[-1])

        for horizon in horizons:
            target_year = train_end_year + horizon
            if target_year not in year_to_row:
                continue

            fit = _fit_and_predict(train_df, target_year, degree)
            actual_row = year_to_row[target_year]

            for target in ("labor", "robot"):
                pred = fit[f"{target}_pred"]
                sigma = fit[f"{target}_sigma"] * np.sqrt(horizon)
                lo = pred - Z_P10_P90 * sigma
                hi = pred + Z_P10_P90 * sigma
                actual = float(getattr(actual_row, "annual_salary" if target == "labor" else "robot_cost"))
                rows.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "actual": actual,
                        "p10": lo,
                        "p90": hi,
                        "covered": 1.0 if lo <= actual <= hi else 0.0,
                        "width": hi - lo,
                    }
                )

    return pd.DataFrame(rows)


def compute_forecast_reliability(
    df: pd.DataFrame,
    degree: int = 2,
    horizons: Tuple[int, ...] = (1, 3, 5),
    min_train_points: int = 8,
) -> Dict[str, object]:
    """Compute interval reliability metrics from rolling-origin backtests."""
    rows = _generate_backtest_rows(df, horizons, degree, min_train_points)
    if rows.empty:
        return {
            "interval_coverage_rate": None,
            "average_interval_width": None,
            "coverage_target": TARGET_INTERVAL_COVERAGE,
            "calibration_by_horizon": {str(h): None for h in horizons},
            "sample_size": 0,
        }

    overall_coverage = float(rows["covered"].mean())
    avg_width = float(rows["width"].mean())

    calibration_by_horizon = {
        str(h): float(rows.loc[rows["horizon"] == h, "covered"].mean())
        if not rows.loc[rows["horizon"] == h].empty
        else None
        for h in horizons
    }

    return {
        "interval_coverage_rate": overall_coverage,
        "average_interval_width": avg_width,
        "coverage_target": TARGET_INTERVAL_COVERAGE,
        "calibration_by_horizon": calibration_by_horizon,
        "sample_size": int(len(rows)),
    }


def save_evaluation_history(
    metrics: Dict[str, object],
    model_version: str,
    history_path: str = DEFAULT_HISTORY_PATH,
) -> pd.DataFrame:
    """Append run metrics to a CSV history keyed by timestamp + model version."""
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    record = {
        "timestamp": timestamp,
        "model_version": model_version,
        "interval_coverage_rate": metrics.get("interval_coverage_rate"),
        "average_interval_width": metrics.get("average_interval_width"),
        "coverage_target": metrics.get("coverage_target", TARGET_INTERVAL_COVERAGE),
        "sample_size": metrics.get("sample_size", 0),
        "calibration_by_horizon": json.dumps(metrics.get("calibration_by_horizon", {})),
    }

    history_df = pd.DataFrame([record])
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        history_df = pd.concat([existing, history_df], ignore_index=True)

    history_df.to_csv(history_path, index=False)
    return history_df


def load_latest_evaluation(history_path: str = DEFAULT_HISTORY_PATH) -> Dict[str, object] | None:
    """Load latest evaluation record from disk."""
    if not os.path.exists(history_path):
        return None

    history_df = pd.read_csv(history_path)
    if history_df.empty:
        return None

    latest = history_df.iloc[-1].to_dict()
    calibration = latest.get("calibration_by_horizon", "{}")
    if isinstance(calibration, str):
        latest["calibration_by_horizon"] = json.loads(calibration)
    return latest
