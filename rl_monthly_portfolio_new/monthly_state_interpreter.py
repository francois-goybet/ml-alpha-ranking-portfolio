"""
monthly_state_interpreter.py  (v2 — fully pre-computed)
========================================================
All monthly data is pre-compiled into slot-aligned numpy matrices at
``__init__`` time.  Every runtime call — ``interpret()``,
``get_month_slot_arrays()``, ``get_universe()`` — is a single numpy
slice or dot product with **zero Python loops and zero pandas lookups**.

Memory footprint: ~420 months × 2,200 slots × 5 columns × 4 bytes ≈ 18 MB.

State Layout (flattened)
-------------------------
    [ features(max_N × n_feat) | weights(max_N + 1) ]

Action-Mask Layout
-------------------
    [ stock_0 … stock_{N-1} | CASH ]   — 1 = investable, 0 = masked
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MonthlyStateInterpreter:

    # --------------------------------------------------------------------- #
    #  Construction — ONE-TIME pre-computation
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        data_df: pd.DataFrame,
        sp500_df: pd.DataFrame,
        feature_cols: list[str],
        return_col: str = "ret",
        mcap_col: str = "market_cap_musd",
        max_universe_size: int = 2200,
        riskfree_df: Optional[pd.DataFrame] = None,
        delisting_df: Optional[pd.DataFrame] = None,
    ):
        self.feature_cols      = list(feature_cols)
        self.return_col        = return_col
        self.mcap_col          = mcap_col
        self.max_universe_size = max_universe_size
        self.n_features        = len(self.feature_cols)

        # Derived dimensions
        self.state_dim = max_universe_size * self.n_features + max_universe_size + 1
        self.mask_dim  = max_universe_size + 1

        # ---- build permno → slot mapping ----
        data_df = data_df.copy()
        data_df["permno"] = data_df["permno"].astype(int)
        data_df["yyyymm"] = data_df["yyyymm"].astype(int)

        all_permnos = sorted(data_df["permno"].unique())
        self._permno_to_slot: Dict[int, int] = {
            p: i for i, p in enumerate(all_permnos)
        }
        self._slot_to_permno: Dict[int, int] = {
            i: p for p, i in self._permno_to_slot.items()
        }
        n_slots = len(all_permnos)
        if n_slots > max_universe_size:
            warnings.warn(
                f"Total unique permnos ({n_slots}) exceeds "
                f"max_universe_size ({max_universe_size}).  Increase it."
            )

        # ---- build month index ----
        all_months = np.sort(data_df["yyyymm"].unique())
        self._all_months = all_months
        self._month_to_idx: Dict[int, int] = {
            int(ym): i for i, ym in enumerate(all_months)
        }
        n_months = len(all_months)

        # ---- determine which columns to pre-compute ----
        cost_cols = ["DolVol", "BidAskSpread", "VolMkt"]
        all_needed: list[str] = []
        seen: set[str] = set()
        # Return column first (index 0)
        all_needed.append(return_col)
        seen.add(return_col)
        # Then cost columns
        for c in cost_cols:
            if c not in seen and c in data_df.columns:
                all_needed.append(c)
                seen.add(c)
        # Then feature columns (may overlap with cost cols)
        for c in feature_cols:
            if c not in seen and c in data_df.columns:
                all_needed.append(c)
                seen.add(c)

        self._col_names = all_needed
        self._col_to_idx = {c: i for i, c in enumerate(all_needed)}
        n_cols = len(all_needed)

        # Feature column indices into the data cube
        self._feat_col_indices = np.array(
            [self._col_to_idx[c] for c in feature_cols if c in self._col_to_idx],
            dtype=int,
        )
        # Cost column indices
        self._ret_idx    = self._col_to_idx[return_col]
        self._dolvol_idx = self._col_to_idx.get("DolVol", -1)
        self._spread_idx = self._col_to_idx.get("BidAskSpread", -1)
        self._volmkt_idx = self._col_to_idx.get("VolMkt", -1)

        # ================================================================
        #  PRE-COMPUTE: data_cube (n_months, max_N, n_cols)
        #               present   (n_months, max_N)
        # ================================================================
        max_N = max_universe_size
        mem_mb = n_months * max_N * n_cols * 4 / 1e6
        print(f"  [Interpreter] Pre-computing {n_months}×{max_N}×{n_cols} "
              f"data cube ({mem_mb:.1f} MB) …")

        data_cube = np.zeros((n_months, max_N, n_cols), dtype=np.float32)
        present   = np.zeros((n_months, max_N), dtype=bool)

        # Build slot and month index columns for vectorised fill
        slot_arr  = data_df["permno"].map(self._permno_to_slot)
        month_arr = data_df["yyyymm"].map(self._month_to_idx)

        # Drop rows where permno or month is unknown or slot >= max_N
        valid = slot_arr.notna() & month_arr.notna()
        valid &= slot_arr < max_N
        slot_arr  = slot_arr[valid].astype(int).values
        month_arr = month_arr[valid].astype(int).values

        # Fill each column with vectorised indexing
        for ci, col in enumerate(all_needed):
            if col not in data_df.columns:
                continue
            vals = data_df.loc[valid.values, col].values.astype(np.float64)
            nan_mask = np.isfinite(vals)
            # Advanced indexing: data_cube[month, slot, col] = val
            data_cube[month_arr[nan_mask], slot_arr[nan_mask], ci] = vals[nan_mask].astype(np.float32)

        # Present = has a valid return value
        ret_vals = data_df.loc[valid.values, return_col].values.astype(np.float64)
        ret_finite = np.isfinite(ret_vals)
        present[month_arr[ret_finite], slot_arr[ret_finite]] = True

        self._data_cube = data_cube
        self._present   = present

        # ================================================================
        #  SAVE RAW COST COLUMNS before Z-scoring mutates data_cube.
        #
        #  The environment's _calculate_costs() and _compute_gross_return()
        #  need the ORIGINAL values of ret, DolVol, BidAskSpread, VolMkt.
        #  If any of these columns are also in feature_cols (they are!),
        #  the Z-score loop below will overwrite them with z-scores,
        #  producing negative spreads/vol → negative transaction costs
        #  → the agent gets PAID to trade.  Fix: snapshot them first.
        # ================================================================
        self._raw_ret    = data_cube[:, :max_N, self._ret_idx].copy()        # (n_months, max_N)
        self._raw_dolvol = (
            data_cube[:, :max_N, self._dolvol_idx].copy()
            if self._dolvol_idx >= 0
            else np.zeros((n_months, max_N), dtype=np.float32)
        )
        self._raw_spread = (
            data_cube[:, :max_N, self._spread_idx].copy()
            if self._spread_idx >= 0
            else np.zeros((n_months, max_N), dtype=np.float32)
        )
        self._raw_volmkt = (
            data_cube[:, :max_N, self._volmkt_idx].copy()
            if self._volmkt_idx >= 0
            else np.zeros((n_months, max_N), dtype=np.float32)
        )

        raw_mb = (self._raw_ret.nbytes + self._raw_dolvol.nbytes
                  + self._raw_spread.nbytes + self._raw_volmkt.nbytes) / 1e6
        print(f"  [Interpreter] Raw cost columns saved ({raw_mb:.1f} MB)")

        # ================================================================
        #  NORMALIZE: cross-sectional z-score on STATE FEATURES only.
        #  This now safely mutates data_cube in-place — the raw cost
        #  values are already preserved in _raw_* arrays above.
        # ================================================================
        for ci in self._feat_col_indices:
            for mi in range(n_months):
                mask = present[mi]             # which slots have data
                if mask.sum() < 2:
                    continue
                vals = data_cube[mi, mask, ci]
                mu   = vals.mean()
                std  = vals.std()
                if std > 1e-8:
                    data_cube[mi, mask, ci] = (vals - mu) / std
                else:
                    data_cube[mi, mask, ci] = 0.0

        print(f"  [Interpreter] Data cube ready. "
              f"{int(present.sum()):,} stock-months populated. "
              f"{len(self._feat_col_indices)} feature cols z-scored.")

        # ---- SP500 benchmark: yyyymm → float ----
        self._sp500: Dict[int, float] = {}
        for _, row in sp500_df.iterrows():
            self._sp500[int(row["yyyymm"])] = float(row["ret"])

        # ---- Risk-free rate: yyyymm → float ----
        self._riskfree: Dict[int, float] = {}
        if riskfree_df is not None and len(riskfree_df) > 0:
            for _, row in riskfree_df.iterrows():
                self._riskfree[int(row["yyyymm"])] = float(row["rf"])

        # ---- Delisting returns: (permno, yyyymm) → float ----
        self._delisting: Dict[tuple[int, int], float] = {}
        if delisting_df is not None and len(delisting_df) > 0:
            for _, row in delisting_df.iterrows():
                key = (int(row["permno"]), int(row["yyyymm"]))
                self._delisting[key] = float(row["dlret"])

    # --------------------------------------------------------------------- #
    #  Public API: Universe Queries
    # --------------------------------------------------------------------- #

    def get_available_months(self, start_ym: int, end_ym: int) -> np.ndarray:
        mask = (self._all_months >= start_ym) & (self._all_months <= end_ym)
        return self._all_months[mask]

    def get_universe(self, yyyymm: int) -> list[int]:
        """Return permnos with valid return data this month."""
        mi = self._month_to_idx.get(int(yyyymm))
        if mi is None:
            return []
        slots = np.where(self._present[mi])[0]
        return [self._slot_to_permno[int(s)] for s in slots
                if int(s) in self._slot_to_permno]

    def get_benchmark_return(self, yyyymm: int) -> float:
        return self._sp500.get(int(yyyymm), np.nan)

    def get_riskfree_rate(self, yyyymm: int) -> float:
        return self._riskfree.get(int(yyyymm), 0.0)

    def get_delisting_return(self, permno: int, yyyymm: int) -> Optional[float]:
        return self._delisting.get((int(permno), int(yyyymm)), None)

    # --------------------------------------------------------------------- #
    #  State Construction  (pure numpy slicing — ZERO Python loops)
    # --------------------------------------------------------------------- #

    def interpret(
        self,
        yyyymm: int,
        active_universe: list[int],
        current_weights: np.ndarray,
        cash_weight: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build observation + action mask via numpy slicing.
        No per-stock Python loops.
        """
        max_N = self.max_universe_size
        mi    = self._month_to_idx.get(int(yyyymm))

        # ---- feature matrix: one numpy slice ----
        if mi is not None:
            feat_matrix = self._data_cube[mi, :max_N, self._feat_col_indices]
        else:
            feat_matrix = np.zeros((max_N, self.n_features), dtype=np.float32)

        # ---- weight vector ----
        weight_vec = np.empty(max_N + 1, dtype=np.float32)
        weight_vec[:max_N] = current_weights[:max_N]
        weight_vec[max_N]  = cash_weight

        # ---- flatten and concatenate ----
        state = np.concatenate([
            feat_matrix.ravel(),
            weight_vec,
        ])

        # Action mask is built by the env's _build_live_mask — return dummy
        action_mask = np.zeros(max_N + 1, dtype=np.float32)
        action_mask[-1] = 1.0

        return state, action_mask

    # --------------------------------------------------------------------- #
    #  Vectorised Slot Arrays  (direct cube slicing)
    # --------------------------------------------------------------------- #

    def get_month_slot_arrays(self, yyyymm: int) -> Dict[str, np.ndarray]:
        """
        Return slot-aligned arrays for the environment's financial math
        (returns, transaction costs).  Uses the RAW pre-Z-score copies
        so that spreads / vol / returns are in their original units.
        Zero allocation, zero loops.
        """
        mi = self._month_to_idx.get(int(yyyymm))
        max_N = self.max_universe_size

        if mi is None:
            z = np.zeros(max_N, dtype=np.float32)
            return {"ret": z, "DolVol": z.copy(), "BidAskSpread": z.copy(),
                    "VolMkt": z.copy(), "present": np.zeros(max_N, dtype=bool)}

        return {
            "ret":          self._raw_ret[mi],
            "DolVol":       self._raw_dolvol[mi],
            "BidAskSpread": self._raw_spread[mi],
            "VolMkt":       self._raw_volmkt[mi],
            "present":      self._present[mi, :max_N],
        }

    def clear_slot_cache(self) -> None:
        """No-op — pre-computed cube needs no cache management."""
        pass

    # --------------------------------------------------------------------- #
    #  Legacy single-stock lookup (kept for sweep_orphans edge case)
    # --------------------------------------------------------------------- #

    def get_stock_data(self, yyyymm: int, permno: int) -> Dict[str, float]:
        mi   = self._month_to_idx.get(int(yyyymm))
        slot = self._permno_to_slot.get(int(permno))
        if mi is None or slot is None or slot >= self.max_universe_size:
            return {}
        if not self._present[mi, slot]:
            return {}
        row = self._data_cube[mi, slot, :]
        return {col: float(row[i]) for i, col in enumerate(self._col_names)}