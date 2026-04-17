#!/usr/bin/env python3
"""
generate_rf_rate.py
====================
Downloads the monthly risk-free rate (1-month T-bill) from the
Kenneth French Data Library and saves it as a parquet file matching
the pipeline's expected format.

NOTE: If you have already run the ranking pipeline, DataManager.get_rf()
produces ``data/rf.parquet`` from WRDS (ff.factors_monthly) with the same
format.  In that case you do NOT need this script — it exists as a
WRDS-free fallback for standalone RL development.

Source
------
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
    → "Fama/French 3 Factors" monthly CSV
    → The RF column is the monthly risk-free rate as a decimal percentage
      (e.g. 0.43 means 0.43% per month, which we convert to 0.0043).

Output Format
--------------
    Parquet with columns:
        yyyymm  (int)    — e.g. 199001
        rf      (float)  — monthly risk-free rate as a decimal fraction
                           (0.0043 = 0.43% per month)

Usage
-----
    python generate_rf_rate.py
    python generate_rf_rate.py --start 1990 --end 2024
    python generate_rf_rate.py --output data/rf.parquet

Alternative: WRDS Download
---------------------------
    If you have WRDS access, you can also get the risk-free rate from the
    CRSP Treasury Fama 1-Month T-Bill series:

        import wrds
        db = wrds.Connection()
        rf = db.raw_sql('''
            SELECT caldt, t30ret as rf
            FROM crsp.mcti
            WHERE caldt >= '1990-01-01' AND caldt <= '2024-12-31'
        ''')
        rf['yyyymm'] = rf['caldt'].dt.year * 100 + rf['caldt'].dt.month
        rf[['yyyymm', 'rf']].to_parquet('data/raw/rf_monthly.parquet')

    The Kenneth French approach below requires no login and no subscription.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# URL for the Fama-French 3-Factor monthly CSV (zipped)
FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_CSV.zip"
)


def download_ff3_factors() -> pd.DataFrame:
    """
    Download the Fama-French 3-Factor file and extract the monthly table.

    The CSV inside the ZIP has a quirky layout:
      - Header lines at the top
      - Monthly data (6-digit YYYYMM in first column)
      - Then an "Annual Factors" section
    We parse only the monthly rows (those with a valid 6-digit date).
    """
    try:
        # Try urllib first (no extra dependencies)
        import urllib.request
        print(f"Downloading from Kenneth French Data Library …")
        response = urllib.request.urlopen(FF3_URL, timeout=30)
        data = response.read()
    except Exception as e:
        print(f"urllib failed ({e}), trying requests …")
        import requests
        resp = requests.get(FF3_URL, timeout=30)
        resp.raise_for_status()
        data = resp.content

    # Extract CSV from ZIP
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")]
        if not csv_name:
            raise ValueError(f"No CSV file found in ZIP. Contents: {zf.namelist()}")
        csv_bytes = zf.read(csv_name[0])

    # Parse the CSV
    # The file has a variable number of header lines, then data lines
    # starting with a 6-digit YYYYMM.  We skip lines until we find
    # numeric data, then stop when we hit the annual section.
    lines = csv_bytes.decode("utf-8", errors="replace").splitlines()

    rows = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        date_str = parts[0].strip()
        # Monthly rows have exactly 6 digits (YYYYMM)
        if len(date_str) == 6 and date_str.isdigit():
            yyyymm = int(date_str)
            try:
                # RF is the 4th column (0-indexed: col 3)
                rf_pct = float(parts[3].strip())
                rows.append({"yyyymm": yyyymm, "rf_pct": rf_pct})
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} monthly observations "
          f"({df['yyyymm'].min()} to {df['yyyymm'].max()})")
    return df


def process_rf(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Filter to the requested date range and convert RF from percentage
    points to a decimal fraction.

    Kenneth French reports RF as e.g. 0.43 meaning 0.43% per month.
    We convert to 0.0043 (decimal fraction).
    """
    start_ym = start_year * 100 + 1
    end_ym   = end_year * 100 + 12

    df = df[(df["yyyymm"] >= start_ym) & (df["yyyymm"] <= end_ym)].copy()
    df["rf"] = df["rf_pct"] / 100.0      # 0.43% → 0.0043
    df = df[["yyyymm", "rf"]].sort_values("yyyymm").reset_index(drop=True)

    print(f"  Filtered to {start_year}–{end_year}: {len(df)} months")
    print(f"  RF range: {df['rf'].min():.6f} to {df['rf'].max():.6f} "
          f"(mean {df['rf'].mean():.6f})")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download monthly risk-free rate from Kenneth French")
    parser.add_argument("--start",  type=int, default=1990)
    parser.add_argument("--end",    type=int, default=2024)
    parser.add_argument("--output", default="data/rf.parquet")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = download_ff3_factors()
    rf  = process_rf(raw, args.start, args.end)

    rf.to_parquet(out_path, index=False)
    print(f"\n✓  Saved to {out_path}")
    print(f"   Columns: {rf.columns.tolist()}")
    print(f"   Shape:   {rf.shape}")

    # Show a few rows as sanity check
    print(f"\n   Sample:")
    print(rf.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
