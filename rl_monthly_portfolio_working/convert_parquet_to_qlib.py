#!/usr/bin/env python3
"""
convert_parquet_to_qlib.py
==========================
Converts a WRDS / CRSP monthly parquet dataset and an SP500 benchmark parquet
into Qlib's binary storage format (monthly frequency).

Qlib Binary Layout
-------------------
    {qlib_dir}/
    ├── calendars/
    │   └── month.txt              # one date per line (YYYY-MM-DD)
    ├── instruments/
    │   └── all.txt                # ticker <TAB> start <TAB> end
    └── features/
        ├── SH_permno_10078/       # one directory per stock
        │   ├── ret.bin            # flat float32 arrays aligned to calendar
        │   ├── DolVol.bin
        │   └── ...
        └── INDEX_SP500/           # benchmark pseudo-ticker
            └── ret.bin

Usage
-----
    python convert_parquet_to_qlib.py                 # defaults from config.yaml
    python convert_parquet_to_qlib.py --alpha          # also merge xgboost_rank_oos
    python convert_parquet_to_qlib.py --config my.yaml # custom config path
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _yyyymm_to_date(yyyymm: int | str) -> pd.Timestamp:
    """Convert 199605 → pd.Timestamp('1996-05-01')."""
    s = str(int(yyyymm))
    return pd.Timestamp(f"{s[:4]}-{s[4:6]}-01")


def _write_bin(path: Path, values: np.ndarray) -> None:
    """Write a 1-D float32 array as a Qlib .bin file (raw little-endian)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype("<f").tofile(str(path))      # little-endian float32


def _write_calendar(cal_path: Path, dates: list[pd.Timestamp]) -> None:
    """Write the calendar file: one YYYY-MM-DD date per line."""
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "w") as f:
        for d in sorted(dates):
            f.write(d.strftime("%Y-%m-%d") + "\n")


def _write_instruments(inst_path: Path, records: list[tuple[str, str, str]]) -> None:
    """Write instruments/all.txt: ticker \\t start \\t end."""
    inst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inst_path, "w") as f:
        for ticker, start, end in sorted(records):
            f.write(f"{ticker}\t{start}\t{end}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def convert(cfg: dict, merge_alpha: bool = False) -> None:
    """
    End-to-end conversion: parquet → Qlib binary.

    Parameters
    ----------
    cfg : dict
        Parsed config.yaml.
    merge_alpha : bool
        If True, also read xgboost_rank_oos.parquet and add it as a feature.
    """
    qlib_dir   = Path(cfg["paths"]["qlib_dir"]).expanduser()
    feat_dir   = qlib_dir / "features"
    cal_path   = qlib_dir / "calendars" / "month.txt"
    inst_path  = qlib_dir / "instruments" / "all.txt"

    # ------------------------------------------------------------------
    #  1.  Read main dataset
    # ------------------------------------------------------------------
    raw_path = Path(cfg["paths"]["raw_parquet"])
    print(f"[1/5] Reading main parquet: {raw_path}")
    df = pd.read_parquet(raw_path)

    # Normalise column types
    df["yyyymm"]  = df["yyyymm"].astype(int)
    df["permno"]  = df["permno"].astype(int)
    df["date"]    = df["yyyymm"].apply(_yyyymm_to_date)

    # ------------------------------------------------------------------
    #  2.  Optionally merge alpha signal
    # ------------------------------------------------------------------
    if merge_alpha:
        alpha_path = Path(cfg["alpha_signal"]["path"])
        alpha_col  = cfg["alpha_signal"]["col_name"]
        if alpha_path.exists():
            print(f"[2/5] Merging alpha signal from {alpha_path}")
            alpha_df = pd.read_parquet(alpha_path)
            alpha_df["yyyymm"] = alpha_df["yyyymm"].astype(int)
            alpha_df["permno"] = alpha_df["permno"].astype(int)
            df = df.merge(alpha_df[["permno", "yyyymm", alpha_col]],
                          on=["permno", "yyyymm"], how="left")
            print(f"    → {alpha_col} coverage: "
                  f"{df[alpha_col].notna().mean():.1%}")
        else:
            print(f"[2/5] Alpha file not found at {alpha_path} — skipping.")
    else:
        print("[2/5] Alpha merge not requested — skipping.")

    # ------------------------------------------------------------------
    #  3.  Read SP500 benchmark and add as pseudo-ticker INDEX_SP500
    # ------------------------------------------------------------------
    sp500_path = Path(cfg["paths"]["sp500_parquet"])
    bench_tick = cfg["features"]["benchmark_ticker"]
    print(f"[3/5] Reading SP500 benchmark: {sp500_path}")

    sp500 = pd.read_parquet(sp500_path)
    sp500["yyyymm"] = sp500["yyyymm"].astype(str).str.strip().astype(int)
    sp500["permno"]  = -1                           # sentinel
    sp500["date"]    = sp500["yyyymm"].apply(_yyyymm_to_date)
    # The benchmark only carries the 'ret' column; other features → NaN
    sp500["_ticker"] = bench_tick

    # ------------------------------------------------------------------
    #  4.  Build calendar & instrument list
    # ------------------------------------------------------------------
    print("[4/5] Building calendar and instrument catalogue …")

    # Calendar = sorted union of all dates in main + sp500
    all_dates = sorted(set(df["date"]).union(set(sp500["date"])))
    _write_calendar(cal_path, all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_cal = len(all_dates)
    print(f"    → Calendar: {all_dates[0].date()} to {all_dates[-1].date()} "
          f"({n_cal} months)")

    # Determine which columns to dump as binary features.
    # We dump EVERY numeric column so the data is fully preserved for
    # future experiments.  The RL env will pick only the columns it needs.
    exclude_cols = {"permno", "yyyymm", "date", "sector", "_ticker"}
    feature_cols = [c for c in df.columns
                    if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"    → {len(feature_cols)} numeric features to dump.")

    # ------------------------------------------------------------------
    #  5.  Write binary files per instrument
    # ------------------------------------------------------------------
    print("[5/5] Writing Qlib binary features …")
    instruments: list[tuple[str, str, str]] = []

    # --- 5a. Individual stocks ---
    grouped = df.groupby("permno")
    total = len(grouped)
    for idx, (permno, gdf) in enumerate(grouped):
        ticker = f"SH_permno_{permno}"
        gdf = gdf.sort_values("date")
        start_dt = gdf["date"].iloc[0]
        end_dt   = gdf["date"].iloc[-1]
        instruments.append((
            ticker,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        ))

        # Map this stock's dates to global calendar indices
        local_indices = gdf["date"].map(date_to_idx).values
        start_idx = date_to_idx[start_dt]
        end_idx   = date_to_idx[end_dt]
        length    = end_idx - start_idx + 1

        for col in feature_cols:
            arr = np.full(length, np.nan, dtype=np.float32)
            offsets = local_indices - start_idx
            vals    = gdf[col].values.astype(np.float32)
            arr[offsets] = vals
            _write_bin(feat_dir / ticker / f"{col}.bin", arr)

        if (idx + 1) % 200 == 0 or idx == total - 1:
            print(f"    → {idx + 1}/{total} instruments written")

    # --- 5b. Benchmark pseudo-ticker ---
    sp500 = sp500.sort_values("date")
    sp_start = sp500["date"].iloc[0]
    sp_end   = sp500["date"].iloc[-1]
    instruments.append((
        bench_tick,
        sp_start.strftime("%Y-%m-%d"),
        sp_end.strftime("%Y-%m-%d"),
    ))
    sp_start_idx = date_to_idx[sp_start]
    sp_end_idx   = date_to_idx[sp_end]
    sp_length    = sp_end_idx - sp_start_idx + 1

    sp_arr = np.full(sp_length, np.nan, dtype=np.float32)
    sp_offsets = sp500["date"].map(date_to_idx).values - sp_start_idx
    sp_arr[sp_offsets] = sp500["ret"].values.astype(np.float32)
    _write_bin(feat_dir / bench_tick / "ret.bin", sp_arr)

    # --- 5c. Write instruments file ---
    _write_instruments(inst_path, instruments)

    print(f"\n✓  Qlib data written to {qlib_dir}")
    print(f"   {len(instruments)} instruments  |  {n_cal} calendar months  |  "
          f"{len(feature_cols)} features")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert WRDS monthly parquet → Qlib binary format.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--alpha", action="store_true",
                        help="Also merge xgboost_rank_oos.parquet")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    convert(cfg, merge_alpha=args.alpha)


if __name__ == "__main__":
    main()
