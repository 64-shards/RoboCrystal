"""Backtest challenger models against champion on rolling splits and manage promotion."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from data_pipeline import run_pipeline
from model_registry import (
    REGISTRY_PATH,
    get_champion_model,
    load_registry,
    make_data_version,
    save_registry,
    upsert_model,
    utc_now_iso,
)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error with zero-safe denominator."""
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def fit_predict(train_df: pd.DataFrame, test_year: int, spec: Dict) -> Dict[str, float]:
    """Train using model spec and predict labor/robot for one test year."""
    years = train_df["year"].values
    labor = train_df["annual_salary"].values
    robot = train_df["robot_cost"].values

    labor_degree = spec["hyperparameters"]["labor_poly_degree"]
    labor_coeffs = np.polyfit(years, labor, labor_degree)
    labor_pred = float(np.polyval(labor_coeffs, test_year))

    robot_family = spec["hyperparameters"]["robot_model"]
    if robot_family == "exp":
        robot_coeffs = np.polyfit(years, np.log(robot), 1)
        robot_pred = float(np.exp(np.polyval(robot_coeffs, test_year)))
    else:
        robot_degree = spec["hyperparameters"]["robot_poly_degree"]
        robot_coeffs = np.polyfit(years, robot, robot_degree)
        robot_pred = float(np.polyval(robot_coeffs, test_year))

    return {"labor_pred": labor_pred, "robot_pred": max(robot_pred, 0.0)}


def rolling_backtest(df: pd.DataFrame, spec: Dict, min_train_years: int) -> Dict[str, float]:
    """Evaluate one-step rolling forecasts from min_train_years onward."""
    ordered = df.sort_values("year").reset_index(drop=True)
    labor_errors: List[float] = []
    robot_errors: List[float] = []

    for split_idx in range(min_train_years, len(ordered)):
        train = ordered.iloc[:split_idx]
        test = ordered.iloc[split_idx]
        preds = fit_predict(train, int(test["year"]), spec)
        labor_errors.append(mape(np.array([test["annual_salary"]]), np.array([preds["labor_pred"]])))
        robot_errors.append(mape(np.array([test["robot_cost"]]), np.array([preds["robot_pred"]])))

    combined = float(np.mean([(l + r) / 2 for l, r in zip(labor_errors, robot_errors)])) if labor_errors else float("inf")
    return {
        "splits": len(labor_errors),
        "labor_mape": float(np.mean(labor_errors)) if labor_errors else float("inf"),
        "robot_mape": float(np.mean(robot_errors)) if robot_errors else float("inf"),
        "combined_mape": combined,
    }


def model_specs() -> List[Dict]:
    """Candidate model registry entries to evaluate."""
    return [
        {
            "id": "labor_poly2_robot_exp1",
            "family": "poly+exp",
            "hyperparameters": {"labor_poly_degree": 2, "robot_model": "exp", "robot_poly_degree": None},
        },
        {
            "id": "labor_poly3_robot_exp1",
            "family": "poly+exp",
            "hyperparameters": {"labor_poly_degree": 3, "robot_model": "exp", "robot_poly_degree": None},
        },
        {
            "id": "labor_poly2_robot_poly2",
            "family": "poly+poly",
            "hyperparameters": {"labor_poly_degree": 2, "robot_model": "poly", "robot_poly_degree": 2},
        },
    ]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load local historical dataset, generating one if needed."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    df = run_pipeline()
    df.to_csv(csv_path, index=False)
    return df


def evaluate_and_promote(args) -> None:
    """Run backtests, compare challenger vs champion, and write registry."""
    df = load_dataset(args.data)
    data_version = make_data_version(df)
    train_range = {"start_year": int(df["year"].min()), "end_year": int(df["year"].max()), "version": data_version}

    registry = load_registry(args.registry)
    registry["promotion_policy"] = {
        "min_relative_improvement": args.min_relative_improvement,
        "max_combined_mape": args.max_combined_mape,
        "minimum_splits": args.min_train_years,
    }

    for spec in model_specs():
        metrics = rolling_backtest(df, spec, args.min_train_years)
        record = {
            "id": spec["id"],
            "family": spec["family"],
            "training_data": train_range,
            "hyperparameters": spec["hyperparameters"],
            "backtest_metrics": metrics,
            "promotion_status": "challenger",
            "evaluated_at": utc_now_iso(),
        }
        upsert_model(registry, record)

    champion = get_champion_model(registry)
    models = sorted(registry["models"], key=lambda m: m["backtest_metrics"]["combined_mape"])
    best = models[0]

    should_promote = False
    if champion is None:
        should_promote = True
        reason = "No existing champion; promoting best backtested model."
    else:
        champion_metric = champion["backtest_metrics"]["combined_mape"]
        best_metric = best["backtest_metrics"]["combined_mape"]
        improvement = (champion_metric - best_metric) / champion_metric if champion_metric > 0 else 0
        should_promote = (
            best["id"] != champion["id"]
            and best["backtest_metrics"]["splits"] >= args.min_train_years
            and best_metric <= args.max_combined_mape
            and improvement >= args.min_relative_improvement
        )
        reason = (
            f"Improvement={improvement:.2%}, best={best_metric:.4f}, champion={champion_metric:.4f}."
            if should_promote
            else f"No promotion: Improvement={improvement:.2%}, best={best_metric:.4f}, champion={champion_metric:.4f}."
        )

    for model in registry["models"]:
        model["promotion_status"] = "challenger"

    if should_promote:
        registry["champion_id"] = best["id"]

    for model in registry["models"]:
        if model["id"] == registry.get("champion_id"):
            model["promotion_status"] = "champion"
            model["promotion_note"] = reason
        elif model["id"] == best["id"]:
            model["promotion_note"] = reason

    save_registry(registry, args.registry)

    print(f"[DONE] Model registry updated at {args.registry}")
    print(f"[INFO] Champion: {registry.get('champion_id')}")
    for model in sorted(registry["models"], key=lambda m: m["backtest_metrics"]["combined_mape"]):
        metric = model["backtest_metrics"]["combined_mape"]
        print(f"  - {model['id']}: combined_mape={metric:.4f} ({model['promotion_status']})")


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest challenger models and promote champion when thresholds are met.")
    parser.add_argument("--data", default="historical_costs.csv", help="Path to historical dataset CSV")
    parser.add_argument("--registry", default=REGISTRY_PATH, help="Path to model registry JSON")
    parser.add_argument("--min-train-years", type=int, default=8, help="Minimum initial training years for rolling backtest")
    parser.add_argument("--min-relative-improvement", type=float, default=0.05, help="Required relative improvement over champion")
    parser.add_argument("--max-combined-mape", type=float, default=0.15, help="Absolute performance gate for promotion")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_and_promote(parse_args())
