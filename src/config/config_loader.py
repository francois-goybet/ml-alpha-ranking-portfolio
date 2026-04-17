from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def parse_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Parse config and apply light defaults.

    This function intentionally does not perform strict schema validation.
    """
    config = raw or {}
    if not isinstance(config, dict):
        raise TypeError("Config root must be a dictionary.")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    pipeline_cfg = config.get("pipeline", {})
    ensemble_cfg = config.get("ensemble", {})
    feature_pipeline_cfg = config.get("feature_pipeline", {})
    strategies = config.get("strategies", [])

    if not isinstance(data_cfg, dict):
        raise TypeError("'data' section must be a dictionary.")
    if not isinstance(model_cfg, dict):
        raise TypeError("'model' section must be a dictionary.")
    if not isinstance(pipeline_cfg, dict):
        raise TypeError("'pipeline' section must be a dictionary.")

    data_cfg.setdefault("data_path", "data/raw.csv")
    pipeline_cfg.setdefault("plots_dir", "generated/plots")
    pipeline_cfg.setdefault("rebuild_dataset", False)

    return {
        "data": data_cfg,
        "model": model_cfg,
        "pipeline": pipeline_cfg,
        "ensemble": ensemble_cfg,
        "feature_pipeline": feature_pipeline_cfg,
        "strategies": strategies,
    }


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config from disk and parse it."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return parse_config(raw)
