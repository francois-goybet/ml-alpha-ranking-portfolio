"""End-to-end training pipeline.

Usage
-----
    python pipeline.py
    python pipeline.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src.config.config_loader import load_config
from src.data.DataManager import DataManager
from src.model.model import MultiHorizonRanker

# Columns that are identifiers / targets — never used as features
_META_COLS = {"permno", "yyyymm", "ret", "ret_1m", "ret_3m", "ret_6m"}
_TARGETS = ["ret_1m", "ret_3m", "ret_6m"]


class Pipeline:
    """Load data, split train/val/test, train a MultiHorizonRanker.

    Parameters
    ----------
    config:
        Full config dict (from load_config). Must contain ``data`` and
        ``model`` sections.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.data_cfg = config.get("data", {})
        self.model_cfg = config.get("model", {})

        self.dm = DataManager(self.data_cfg)

        self.df_: pd.DataFrame | None = None
        self.ranker_: MultiHorizonRanker | None = None
        self.feature_cols_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Pull or reload the dataset via DataManager."""
        self.df_ = self.dm.get_data(
            start=self.data_cfg.get("train_start", "1990-01-01"),
            end=self.data_cfg.get("test_end", "2023-12-31"),
            market_cap=self.data_cfg.get("market_cap", 100),
        )
        self.feature_cols_ = [c for c in self.df_.columns if c not in _META_COLS]
        print(f"Dataset loaded — {len(self.df_):,} rows, {len(self.feature_cols_)} features.")
        return self.df_

    def train(self, verbose: bool = False) -> MultiHorizonRanker:
        """Split train/val, build groups, fit the MultiHorizonRanker."""
        if self.df_ is None:
            self.load_data()

        df_train = self._split("train_start", "train_end")
        df_val = self._split("val_start", "val_end")

        X_train, Y_train, groups_train = self._prepare(df_train)
        X_val, Y_val, groups_val = self._prepare(df_val)

        self.ranker_ = MultiHorizonRanker(
            targets=_TARGETS,
            backend="lightgbm",
            num_rounds=self.model_cfg.get("num_rounds", 100),
            learning_rate=self.model_cfg.get("learning_rate", 0.1),
            max_depth=self.model_cfg.get("max_depth", 5),
            subsample=self.model_cfg.get("subsample", 0.8),
            colsample_bytree=self.model_cfg.get("colsample_bytree", 0.8),
            random_state=self.model_cfg.get("random_state", 42),
            verbosity=self.model_cfg.get("verbosity", -1),
        )

        print("Training MultiHorizonRanker …")
        self.ranker_.fit(
            X_train,
            Y_train,
            groups=groups_train,
            eval_set=(X_val, Y_val),
            eval_groups=groups_val,
            verbose=verbose,
        )
        print("Training complete.")
        return self.ranker_

    def predict(self, split: str = "test") -> dict[str, np.ndarray]:
        """Return ranking scores for a given split ('train', 'val', 'test')."""
        if self.ranker_ is None:
            raise ValueError("Call train() first.")

        start_key = f"{split}_start"
        end_key = f"{split}_end"
        df_split = self._split(start_key, end_key)
        X, _, _ = self._prepare(df_split)

        scores = self.ranker_.predict(X)
        print(f"Predictions generated for '{split}' split — {len(X):,} rows.")
        return scores

    def feature_importance(self) -> dict[str, dict[str, float]]:
        """Return feature importances per horizon."""
        if self.ranker_ is None:
            raise ValueError("Call train() first.")
        return self.ranker_.get_feature_importance()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split(self, start_key: str, end_key: str) -> pd.DataFrame:
        """Filter df_ to rows within [start_key, end_key] dates from config."""
        start = int(str(self.data_cfg.get(start_key, "19900101")).replace("-", "")[:6])
        end = int(str(self.data_cfg.get(end_key, "20241231")).replace("-", "")[:6])
        mask = (self.df_["yyyymm"] >= start) & (self.df_["yyyymm"] <= end)
        df = self.df_.loc[mask].sort_values(["yyyymm", "permno"]).reset_index(drop=True)
        print(f"  Split [{start} → {end}]: {len(df):,} rows.")
        return df

    def _prepare(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
        """Return (X, Y, groups) ready for the ranker.

        - X: feature matrix (NaN-filled with 0).
        - Y: target DataFrame with columns ret_1m, ret_3m, ret_6m.
        - groups: list of stock counts per yyyymm (must sum to len(df)).
        """
        X = df[self.feature_cols_].fillna(0.0)
        Y = df[_TARGETS].fillna(0.0)
        groups = df.groupby("yyyymm").size().tolist()
        return X, Y, groups


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    pipe = Pipeline(config)
    pipe.load_data()
    pipe.train(verbose=False)

    importance = pipe.feature_importance()
    for target, imp in importance.items():
        top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop-5 features for {target}:")
        for feat, score in top5:
            print(f"  {feat}: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML ranking pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args)
