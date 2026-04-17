#!/usr/bin/env python3
"""
prepare_delisting_returns.py
=============================
Extracts CRSP delisting returns and saves them in the format expected by
the RL portfolio pipeline.

CRSP provides delisting returns (``dlret``) when a stock is removed from
the exchange.  These capture the terminal value impact:
  • **Merger / acquisition**: often positive (acquisition premium)
  • **Bankruptcy / failure**: often −100% or close to it
  • **Other** (moved to OTC, regulatory, etc.): variable

Without delisting returns, the RL environment assumes a 100% loss for
every stock that disappears — which is too pessimistic for mergers and
too optimistic to distinguish from genuine failures.

Output Format
--------------
    Parquet with columns:
        permno   (int)    — CRSP permanent stock number
        yyyymm   (int)    — last month the stock was present (e.g. 200803)
        dlret    (float)  — delisting return as a decimal fraction
                            (−1.0 = total loss, +0.30 = 30% premium)
        dlstcd   (int)    — CRSP delisting code (for diagnostics)

Usage
-----
    # Option A: From WRDS directly (requires wrds Python package + account)
    python prepare_delisting_returns.py --source wrds

    # Option B: From a pre-downloaded CRSP delisting file
    python prepare_delisting_returns.py --source file --input crsp_delist.csv

    # Option C: Generate a synthetic file from your existing data
    #           (marks all vanishing stocks as −100% loss — same as no file)
    python prepare_delisting_returns.py --source synthetic --main-data data/raw/wrds_monthly.parquet

WRDS SQL Query (for reference)
-------------------------------
    If you prefer to run this manually in WRDS web query:

    SELECT permno,
           EXTRACT(YEAR FROM dlstdt)*100 + EXTRACT(MONTH FROM dlstdt) AS yyyymm,
           dlret,
           dlstcd
    FROM crsp.msedelist
    WHERE dlstdt >= '1990-01-01'
      AND dlstdt <= '2024-12-31'
      AND dlret IS NOT NULL
    ORDER BY permno, dlstdt;

    Save the result as CSV and use --source file --input <path>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#  Source A: WRDS Direct Download
# ═══════════════════════════════════════════════════════════════════════════

def from_wrds(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    """
    Download delisting returns directly from WRDS using the wrds package.
    Requires: pip install wrds  (and a valid WRDS account).
    """
    try:
        import wrds
    except ImportError:
        print("ERROR: 'wrds' package not installed.  Run: pip install wrds")
        print("       Or use --source file with a pre-downloaded CSV.")
        sys.exit(1)

    print("Connecting to WRDS …")
    db = wrds.Connection()

    query = f"""
        SELECT permno,
               EXTRACT(YEAR FROM dlstdt)*100 + EXTRACT(MONTH FROM dlstdt) AS yyyymm,
               dlret,
               dlstcd
        FROM crsp.msedelist
        WHERE dlstdt >= '{start_year}-01-01'
          AND dlstdt <= '{end_year}-12-31'
          AND dlret IS NOT NULL
        ORDER BY permno, dlstdt
    """
    print("Querying crsp.msedelist …")
    df = db.raw_sql(query)
    db.close()

    df["permno"] = df["permno"].astype(int)
    df["yyyymm"] = df["yyyymm"].astype(int)
    df["dlstcd"] = df["dlstcd"].astype(int)
    df["dlret"]  = df["dlret"].astype(float)

    print(f"  Retrieved {len(df)} delisting events")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Source B: Pre-Downloaded CSV
# ═══════════════════════════════════════════════════════════════════════════

def from_file(path: str) -> pd.DataFrame:
    """
    Read a CSV or parquet file containing delisting returns.
    Expected columns: permno, yyyymm (or dlstdt), dlret, dlstcd.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # Handle either yyyymm or dlstdt column
    if "yyyymm" not in df.columns and "dlstdt" in df.columns:
        df["dlstdt"] = pd.to_datetime(df["dlstdt"])
        df["yyyymm"] = df["dlstdt"].dt.year * 100 + df["dlstdt"].dt.month

    required = {"permno", "yyyymm", "dlret"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"       Available: {df.columns.tolist()}")
        sys.exit(1)

    df["permno"] = df["permno"].astype(int)
    df["yyyymm"] = df["yyyymm"].astype(int)
    df["dlret"]  = pd.to_numeric(df["dlret"], errors="coerce")

    # Drop rows where dlret is NaN
    before = len(df)
    df = df.dropna(subset=["dlret"])
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing dlret")

    if "dlstcd" not in df.columns:
        df["dlstcd"] = 0  # unknown

    df["dlstcd"] = df["dlstcd"].astype(int)

    print(f"  Loaded {len(df)} delisting events from {p}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Source C: Synthetic (from existing main data)
# ═══════════════════════════════════════════════════════════════════════════

def from_synthetic(main_data_path: str) -> pd.DataFrame:
    """
    Identify stocks that disappear from the main dataset and create
    synthetic delisting entries.

    This is a FALLBACK: every vanishing stock gets dlret = −1.0
    (total loss), which is the same as having no delisting file at all.
    Use this only as a placeholder while you obtain real CRSP data.
    """
    print(f"Loading main data to detect delistings: {main_data_path}")
    df = pd.read_parquet(main_data_path)
    df["permno"] = df["permno"].astype(int)
    df["yyyymm"] = df["yyyymm"].astype(int)

    # For each stock, find its last month
    last_month = df.groupby("permno")["yyyymm"].max().reset_index()
    last_month.columns = ["permno", "yyyymm"]

    # The overall last month of the dataset — stocks ending here are
    # NOT delistings (they're just the end of the sample).
    max_month = df["yyyymm"].max()
    delistings = last_month[last_month["yyyymm"] < max_month].copy()

    delistings["dlret"]  = -1.0     # worst-case assumption
    delistings["dlstcd"] = 0        # unknown

    print(f"  Identified {len(delistings)} potential delistings "
          f"(all assigned dlret = −1.0)")
    print(f"  NOTE: Replace with real CRSP delisting returns for accuracy.")
    return delistings


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def print_diagnostics(df: pd.DataFrame) -> None:
    """Print summary statistics about the delisting returns."""
    print(f"\n{'─' * 60}")
    print(f"  Delisting Return Diagnostics")
    print(f"{'─' * 60}")
    print(f"  Total events:      {len(df):,}")
    print(f"  Unique stocks:     {df['permno'].nunique():,}")
    print(f"  Date range:        {df['yyyymm'].min()} – {df['yyyymm'].max()}")
    print(f"\n  dlret distribution:")
    print(f"    Mean:            {df['dlret'].mean():+.4f}")
    print(f"    Median:          {df['dlret'].median():+.4f}")
    print(f"    Min:             {df['dlret'].min():+.4f}")
    print(f"    Max:             {df['dlret'].max():+.4f}")
    print(f"    Std:             {df['dlret'].std():.4f}")

    # Breakdown by outcome type
    total_loss = (df["dlret"] <= -0.99).sum()
    negative   = ((df["dlret"] < 0) & (df["dlret"] > -0.99)).sum()
    zero       = (df["dlret"] == 0).sum()
    positive   = (df["dlret"] > 0).sum()

    print(f"\n  Outcome breakdown:")
    print(f"    Total loss (≤ −99%):  {total_loss:>5}  "
          f"({100*total_loss/len(df):.1f}%)")
    print(f"    Partial loss (< 0%):  {negative:>5}  "
          f"({100*negative/len(df):.1f}%)")
    print(f"    Zero return:          {zero:>5}  "
          f"({100*zero/len(df):.1f}%)")
    print(f"    Positive (merger?):   {positive:>5}  "
          f"({100*positive/len(df):.1f}%)")

    if "dlstcd" in df.columns and df["dlstcd"].ne(0).any():
        print(f"\n  Top delisting codes:")
        top_codes = df["dlstcd"].value_counts().head(10)
        for code, count in top_codes.items():
            print(f"    {code:>5}: {count:>5} events")

    print(f"{'─' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare CRSP delisting returns for the RL pipeline")
    parser.add_argument("--source", choices=["wrds", "file", "synthetic"],
                        default="wrds",
                        help="Data source: wrds (direct), file (CSV/parquet), "
                             "or synthetic (fallback)")
    parser.add_argument("--input",  default=None,
                        help="Input file path (for --source file)")
    parser.add_argument("--main-data", default="data/dataset.parquet",
                        help="Main dataset path (for --source synthetic)")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end",   type=int, default=2024)
    parser.add_argument("--output", default="data/delisting_returns.parquet")
    args = parser.parse_args()

    if args.source == "wrds":
        df = from_wrds(args.start, args.end)
    elif args.source == "file":
        if not args.input:
            print("ERROR: --input required with --source file")
            sys.exit(1)
        df = from_file(args.input)
    elif args.source == "synthetic":
        df = from_synthetic(args.main_data)
    else:
        sys.exit(1)

    # Ensure correct column types
    df = df[["permno", "yyyymm", "dlret", "dlstcd"]].copy()
    df["permno"] = df["permno"].astype(int)
    df["yyyymm"] = df["yyyymm"].astype(int)

    print_diagnostics(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✓  Saved to {out_path}")


if __name__ == "__main__":
    main()
