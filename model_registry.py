"""Utilities for reading/writing the model registry artifact."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "model_registry.json")


def utc_now_iso() -> str:
    """Return a UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_data_version(df) -> str:
    """Build a lightweight version fingerprint from year and key metric columns."""
    fingerprint = (
        f"rows={len(df)}|year_min={int(df['year'].min())}|year_max={int(df['year'].max())}|"
        f"labor_sum={float(df['annual_salary'].sum()):.2f}|robot_sum={float(df['robot_cost'].sum()):.2f}"
    )
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:10]


def default_registry() -> Dict[str, Any]:
    """Initial empty registry structure."""
    return {
        "updated_at": utc_now_iso(),
        "champion_id": None,
        "promotion_policy": {
            "min_relative_improvement": 0.05,
            "max_combined_mape": 0.15,
            "minimum_splits": 3,
        },
        "models": [],
    }


def load_registry(path: str = REGISTRY_PATH) -> Dict[str, Any]:
    """Load registry from disk or create a default skeleton."""
    if not os.path.exists(path):
        return default_registry()
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(registry: Dict[str, Any], path: str = REGISTRY_PATH) -> None:
    """Persist the registry to disk with stable formatting."""
    registry["updated_at"] = utc_now_iso()
    with open(path, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)
        file.write("\n")


def get_champion_model(registry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return current champion model metadata."""
    champion_id = registry.get("champion_id")
    for model in registry.get("models", []):
        if model.get("id") == champion_id:
            return model
    return None


def upsert_model(registry: Dict[str, Any], model_record: Dict[str, Any]) -> None:
    """Insert or replace a model record by id."""
    models: List[Dict[str, Any]] = registry.setdefault("models", [])
    for idx, existing in enumerate(models):
        if existing.get("id") == model_record.get("id"):
            models[idx] = model_record
            return
    models.append(model_record)
