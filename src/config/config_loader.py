from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigValidationError(ValueError):
    """Raised when the YAML configuration does not match the expected schema."""


def _expect_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"'{name}' must be a mapping/dictionary.")
    return value


def _assert_required_keys(section_name: str, section: Dict[str, Any], required_keys: set[str]) -> None:
    missing = required_keys - set(section.keys())
    if missing:
        raise ConfigValidationError(
            f"Missing required keys in '{section_name}': {sorted(missing)}"
        )


def _assert_no_unknown_keys(section_name: str, section: Dict[str, Any], allowed_keys: set[str]) -> None:
    unknown = set(section.keys()) - allowed_keys
    if unknown:
        raise ConfigValidationError(
            f"Unknown keys in '{section_name}': {sorted(unknown)}"
        )


def _expect_type(section_name: str, key: str, value: Any, expected_types: tuple[type, ...]) -> None:
    if not isinstance(value, expected_types):
        expected_names = ", ".join(t.__name__ for t in expected_types)
        raise ConfigValidationError(
            f"'{section_name}.{key}' must be of type: {expected_names}. Got: {type(value).__name__}"
        )


def _normalize_date_like(value: Any, field_name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    raise ConfigValidationError(
        f"'{field_name}' must be a string or date value (YYYY-MM-DD)."
    )


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the configuration dictionary.

    This validator is strict:
    - required sections and keys must exist
    - unknown keys are rejected
    - value types are checked
    - date-like values are normalized to ISO strings
    """
    _expect_mapping(config, "config")

    allowed_top_level = {"data_manager", "model", "pipeline"}
    required_top_level = {"data_manager", "model"}
    _assert_required_keys("config", config, required_top_level)
    _assert_no_unknown_keys("config", config, allowed_top_level)

    data_cfg = _expect_mapping(config["data_manager"], "data_manager")
    model_cfg = _expect_mapping(config["model"], "model")
    pipeline_cfg = _expect_mapping(config.get("pipeline", {}), "pipeline")

    data_allowed = {
        "data_path",
        "date_column",
        "target_col",
        "convert_target_to_rank",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
        "exclude_cols",
    }
    data_required = {
        "data_path",
        "date_column",
        "target_col",
        "convert_target_to_rank",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    }
    _assert_required_keys("data_manager", data_cfg, data_required)
    _assert_no_unknown_keys("data_manager", data_cfg, data_allowed)

    _expect_type("data_manager", "data_path", data_cfg["data_path"], (str,))
    _expect_type("data_manager", "date_column", data_cfg["date_column"], (str,))
    _expect_type("data_manager", "target_col", data_cfg["target_col"], (str,))
    _expect_type("data_manager", "convert_target_to_rank", data_cfg["convert_target_to_rank"], (bool,))

    for date_key in ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]:
        data_cfg[date_key] = _normalize_date_like(data_cfg[date_key], f"data_manager.{date_key}")

    if "exclude_cols" in data_cfg:
        _expect_type("data_manager", "exclude_cols", data_cfg["exclude_cols"], (list,))
        for idx, col_name in enumerate(data_cfg["exclude_cols"]):
            _expect_type("data_manager", f"exclude_cols[{idx}]", col_name, (str,))

    model_allowed = {
        "objective",
        "num_rounds",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "random_state",
        "ndcg_exp_gain",
        "verbosity",
        "params",
    }
    model_required = {
        "objective",
        "num_rounds",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "random_state",
        "ndcg_exp_gain",
        "verbosity",
        "params",
    }
    _assert_required_keys("model", model_cfg, model_required)
    _assert_no_unknown_keys("model", model_cfg, model_allowed)

    _expect_type("model", "objective", model_cfg["objective"], (str,))
    _expect_type("model", "num_rounds", model_cfg["num_rounds"], (int,))
    _expect_type("model", "learning_rate", model_cfg["learning_rate"], (float, int))
    _expect_type("model", "max_depth", model_cfg["max_depth"], (int,))
    _expect_type("model", "subsample", model_cfg["subsample"], (float, int))
    _expect_type("model", "colsample_bytree", model_cfg["colsample_bytree"], (float, int))
    _expect_type("model", "random_state", model_cfg["random_state"], (int,))
    _expect_type("model", "ndcg_exp_gain", model_cfg["ndcg_exp_gain"], (bool,))
    _expect_type("model", "verbosity", model_cfg["verbosity"], (int,))
    _expect_type("model", "params", model_cfg["params"], (dict,))

    pipeline_allowed = {"plots_dir", "top_k_preview"}
    _assert_no_unknown_keys("pipeline", pipeline_cfg, pipeline_allowed)
    if "plots_dir" in pipeline_cfg:
        _expect_type("pipeline", "plots_dir", pipeline_cfg["plots_dir"], (str,))
    if "top_k_preview" in pipeline_cfg:
        _expect_type("pipeline", "top_k_preview", pipeline_cfg["top_k_preview"], (int,))

    return {
        "data_manager": data_cfg,
        "model": model_cfg,
        "pipeline": pipeline_cfg,
    }


def load_and_validate_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file from disk and validate its schema."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if raw is None:
        raise ConfigValidationError("Config file is empty.")

    return validate_config(raw)
