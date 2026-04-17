#!/usr/bin/env python3
"""
export_alpha_signal.py
=======================
Converts the ranking pipeline's ensemble predictions into the alpha signal
format consumed by the RL portfolio pipeline.

The ranking pipeline (main.py) saves:
    generated/X_test.csv   — test-set features (includes permno, yyyymm)
    generated/y_pred.csv   — ensemble predicted ranks

This script reads those files, computes a cross-sectional rank ∈ [0, 1]
per month, and saves the result as a parquet that the RL pipeline can
pick up automatically.

Usage
-----
    # After running the ranking pipeline:
    python export_alpha_signal.py

    # Custom paths:
    python export_alpha_signal.py \
        --x-test generated/X_test.csv \
        --y-pred generated/y_pred.csv \
        --output data/xgboost_rank_oos.parquet

    # From a full walk-forward run (all periods, not just test):
    python export_alpha_signal.py \
        --x-test generated/X_all.csv \
        --y-pred generated/y_pred_all.csv

Walk-Forward Note
------------------
    The ranking pipeline's main.py only produces predictions for the TEST
    period by default.  For the RL pipeline's training period to include the
    alpha signal, you need to run the ranker in walk-forward (expanding
    window) mode across all periods.  Until then, the RL agent trains with
    3 features only (DolVol, BidAskSpread, VolMkt) for months outside the
    ranking pipeline's test window — it handles this gracefully.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def export(
    x_test_path: str,
    y_pred_path: str,
    output_path: str,
    pred_col: str = "0",
) -> None:
    """
    Read X_test and y_pred, compute cross-sectional rank, and save.

    Parameters
    ----------
    x_test_path : str
        Path to the CSV containing test features (must have permno, yyyymm).
    y_pred_path : str
        Path to the CSV containing ensemble predictions.
    output_path : str
        Where to save the alpha signal parquet.
    pred_col : str
        Column name in y_pred CSV to use as the raw score.
        Default "0" (pandas default when saving a Series/array to CSV).
    """
    print(f"Loading X_test from {x_test_path} …")
    X = pd.read_csv(x_test_path)

    print(f"Loading y_pred from {y_pred_path} …")
    y_pred = pd.read_csv(y_pred_path)

    # ---- Identify the prediction column ----
    # The ranking pipeline saves y_pred as a DataFrame with a single column.
    # The column name might be "0", "y_pred", or the first column.
    if pred_col in y_pred.columns:
        scores = y_pred[pred_col].values
    else:
        # Fall back to the first (and likely only) column
        scores = y_pred.iloc[:, 0].values
        print(f"  Using column '{y_pred.columns[0]}' as prediction scores")

    if len(scores) != len(X):
        raise ValueError(
            f"Length mismatch: X_test has {len(X)} rows but y_pred has "
            f"{len(scores)} rows."
        )

    # ---- Build the alpha signal DataFrame ----
    df = pd.DataFrame({
        "permno": X["permno"].astype(int),
        "yyyymm": X["yyyymm"].astype(int),
        "raw_score": scores.astype(float),
    })

    # ---- Cross-sectional rank ∈ [0, 1] per month ----
    # Higher raw_score → higher rank → higher alpha.
    # pct=True gives a value in (0, 1] where 1.0 = best stock that month.
    df["xgboost_rank_oos"] = (
        df.groupby("yyyymm")["raw_score"]
          .rank(method="average", ascending=True, pct=True)
    )

    # Drop the raw score — the RL pipeline only needs the normalised rank
    result = df[["permno", "yyyymm", "xgboost_rank_oos"]].copy()

    # ---- Diagnostics ----
    n_months = result["yyyymm"].nunique()
    n_stocks = result["permno"].nunique()
    ym_range = f"{result['yyyymm'].min()} – {result['yyyymm'].max()}"

    print(f"\n  Alpha signal summary:")
    print(f"    Months:        {n_months}")
    print(f"    Unique stocks: {n_stocks}")
    print(f"    Date range:    {ym_range}")
    print(f"    Rank stats:    mean={result['xgboost_rank_oos'].mean():.4f}, "
          f"std={result['xgboost_rank_oos'].std():.4f}")

    # ---- Save ----
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"\n✓  Saved to {out}")
    print(f"   The RL pipeline will auto-detect this file and use it as a 4th state feature.")


def main():
    parser = argparse.ArgumentParser(
        description="Export ranking pipeline predictions → RL alpha signal")
    parser.add_argument("--x-test",  default="../generated/X_test.csv",
                        help="Path to X_test CSV from ranking pipeline")
    parser.add_argument("--y-pred",  default="../generated/y_pred.csv",
                        help="Path to y_pred CSV from ranking pipeline")
    parser.add_argument("--output",  default="data/xgboost_rank_oos.parquet",
                        help="Output parquet path for RL pipeline")
    parser.add_argument("--pred-col", default="0",
                        help="Column name in y_pred to use as raw score")
    args = parser.parse_args()

    export(args.x_test, args.y_pred, args.output, args.pred_col)


if __name__ == "__main__":
    main()
