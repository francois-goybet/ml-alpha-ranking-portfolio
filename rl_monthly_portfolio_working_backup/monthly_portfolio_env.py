"""
monthly_portfolio_env.py
========================
Gymnasium environment for monthly portfolio allocation via Deep RL.

Key Design Decisions
---------------------
1.  **Episodic**: Each episode spans ``episode_length`` months (default 36).
2.  **Stock Dropout**: On ``reset()``, a configurable fraction of the valid
    universe is randomly dropped.  The surviving ``active_universe`` is locked
    for the entire episode, simulating real-world portfolio constraints.
3.  **Orphan Resolution (Sweep-to-Cash)**:
      • Stock dropped by *our* artificial dropout  → weight swept to cash at 0% loss.
      • Stock *actually delisted* in the data       → weight swept to cash at 100% loss.
4.  **Transaction Costs**: half-spread cost + Square-Root-Law market impact.
5.  **Action Masking**: The observation dict includes an ``action_mask`` so
    the policy network can zero-out logits for inactive slots before softmax.
6.  **Reward**: Net portfolio return minus SP500 benchmark return (Active Return).

Action Space
-------------
    Box(shape=(max_universe_size + 1,))  — raw logits.
    The environment applies masking + softmax internally to produce valid
    portfolio weights that sum to 1.

Observation Space
------------------
    Dict({
        "obs":         Box(shape=(state_dim,)),
        "action_mask": Box(shape=(max_universe_size + 1,), low=0, high=1),
    })
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from monthly_state_interpreter import MonthlyStateInterpreter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: Masked Softmax
# ═══════════════════════════════════════════════════════════════════════════

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply action mask then softmax.

    Masked positions receive -1e9 before exponentiation so their resulting
    weight is effectively 0.

    Parameters
    ----------
    logits : (K,)   raw action outputs from the policy network.
    mask   : (K,)   1 = valid, 0 = masked.

    Returns
    -------
    weights : (K,)  non-negative, sums to 1.
    """
    masked = np.where(mask > 0.5, logits, -1e9)
    # Numerical stability: subtract max before exp
    shifted = masked - masked.max()
    exp_vals = np.exp(shifted)
    total = exp_vals.sum()
    if total < 1e-12:
        # Fallback: if everything is masked (shouldn't happen — cash is
        # always active), put 100 % in cash (last slot).
        weights = np.zeros_like(logits)
        weights[-1] = 1.0
        return weights
    return exp_vals / total


# ═══════════════════════════════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════════════════════════════

class MonthlyPortfolioEnv(gym.Env):
    """
    Monthly portfolio allocation environment.

    Parameters
    ----------
    interpreter : MonthlyStateInterpreter
        Pre-loaded data handler.
    config : dict
        ``env`` section of config.yaml, plus ``features`` section.
    mode : str
        ``"train"``  → episodes sample random start months.
        ``"eval"``   → episodes march sequentially from ``eval_start``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        interpreter: MonthlyStateInterpreter,
        config: dict,
        mode: str = "train",
    ):
        super().__init__()

        self.interp   = interpreter
        self.cfg      = config
        self.mode     = mode

        # ---- unpack config --------------------------------------------------
        env_cfg = config["env"]
        self.episode_length   = int(env_cfg["episode_length"])
        self.max_N            = int(env_cfg["max_universe_size"])
        self.dropout_rate     = float(env_cfg["dropout_rate"])
        self.initial_capital  = float(env_cfg["initial_capital"])
        self.init_weight_mode = str(env_cfg.get("initial_weights", "cash"))
        self.impact_eta       = float(env_cfg["cost"]["impact_eta"])
        self.bench_fallback   = float(env_cfg["reward"].get("benchmark_missing", 0.0))

        # Risk-free rate: applied to cash holdings each month
        rf_cfg = env_cfg.get("riskfree", {})
        self.use_riskfree     = bool(rf_cfg.get("enabled", False))
        self.rf_annualise     = bool(rf_cfg.get("annualise", False))

        # Delisting returns: use actual dlret instead of -100% loss
        dl_cfg = env_cfg.get("delisting", {})
        self.use_delisting    = bool(dl_cfg.get("enabled", False))

        # ---- date range for episode sampling --------------------------------
        dates_cfg  = config["dates"]
        if mode == "train":
            start_ym = int(dates_cfg["train_start"].replace("-", ""))
            end_ym   = int(dates_cfg["train_end"].replace("-", ""))
        elif mode == "val":
            start_ym = int(dates_cfg["val_start"].replace("-", ""))
            end_ym   = int(dates_cfg["val_end"].replace("-", ""))
        else:  # test
            start_ym = int(dates_cfg["test_start"].replace("-", ""))
            end_ym   = int(dates_cfg["test_end"].replace("-", ""))

        self._avail_months = interpreter.get_available_months(start_ym, end_ym)
        # We need at least episode_length consecutive months for an episode.
        # The start month can be at most (len - episode_length) into the array.
        self._max_start_idx = max(0, len(self._avail_months) - self.episode_length)

        # ---- spaces ---------------------------------------------------------
        state_dim = interpreter.state_dim
        mask_dim  = interpreter.mask_dim       # max_N + 1

        self.observation_space = spaces.Dict({
            "obs":         spaces.Box(-np.inf, np.inf, shape=(state_dim,),
                                      dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(mask_dim,),
                                      dtype=np.float32),
        })
        # SB3 requires finite action-space bounds.  Since we apply masked
        # softmax in step(), the raw logit scale is irrelevant — softmax is
        # shift-invariant.  [-10, 10] is more than sufficient; any logit
        # beyond ±5 already saturates the softmax to ~0 or ~1.
        self.action_space = spaces.Box(-10.0, 10.0, shape=(mask_dim,),
                                       dtype=np.float32)

        # ---- permno ↔ slot mappings (mirrors interpreter) -------------------
        self._permno_to_slot = interpreter._permno_to_slot
        self._slot_to_permno = {v: k for k, v in self._permno_to_slot.items()}

        # ---- episode state (initialised in reset) ---------------------------
        self._step_idx:        int = 0
        self._episode_months:  np.ndarray = np.array([], dtype=int)
        self._active_universe: list[int] = []
        self._active_slots:    set[int]  = set()
        self._weights:         np.ndarray = np.zeros(self.max_N, dtype=np.float32)
        self._cash:            float = 1.0
        self._portfolio_value: float = self.initial_capital

    # ================================================================== #
    #  reset()
    # ================================================================== #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Start a new episode.

        1. Sample a random contiguous block of ``episode_length`` months.
        2. Identify the valid stock universe for the first month.
        3. Apply Stock Dropout: randomly remove ``dropout_rate`` of stocks.
        4. Initialise portfolio to 100 % cash (or alternative mode).
        """
        super().reset(seed=seed)

        # ---- 1. sample start ------------------------------------------------
        if self.mode == "train":
            start_idx = self.np_random.integers(0, self._max_start_idx + 1)
        else:
            # For eval/test: sequential, optionally overridden via options
            start_idx = (options or {}).get("start_idx", 0)

        self._episode_months = self._avail_months[
            start_idx : start_idx + self.episode_length
        ]
        if len(self._episode_months) < self.episode_length:
            # Edge case: not enough months left — wrap or truncate
            self._episode_months = self._avail_months[-self.episode_length:]

        first_month = int(self._episode_months[0])

        # ---- 2. valid universe ----------------------------------------------
        full_universe = self.interp.get_universe(first_month)
        n_full = len(full_universe)

        # ---- 3. stock dropout -----------------------------------------------
        n_drop = int(np.floor(n_full * self.dropout_rate))
        if n_drop > 0 and self.mode == "train":
            drop_idx = self.np_random.choice(n_full, size=n_drop, replace=False)
            keep_mask = np.ones(n_full, dtype=bool)
            keep_mask[drop_idx] = False
            self._active_universe = [
                full_universe[i] for i in range(n_full) if keep_mask[i]
            ]
        else:
            # No dropout during evaluation
            self._active_universe = list(full_universe)

        self._active_slots = {
            self._permno_to_slot[p]
            for p in self._active_universe
            if p in self._permno_to_slot
        }

        # Boolean mask for vectorised operations (True for active slots)
        self._active_slots_mask = np.zeros(self.max_N, dtype=bool)
        for s in self._active_slots:
            if s < self.max_N:
                self._active_slots_mask[s] = True

        # Clear the interpreter's slot-array cache for the new episode
        self.interp.clear_slot_cache()

        logger.debug(
            f"Episode reset: month={first_month}, "
            f"universe={n_full}→{len(self._active_universe)} "
            f"(dropped {n_drop})"
        )

        # ---- 4. initial portfolio -------------------------------------------
        self._weights = np.zeros(self.max_N, dtype=np.float32)
        self._cash    = 1.0
        self._portfolio_value = self.initial_capital
        self._step_idx = 0

        if self.init_weight_mode == "equal":
            # Future extension: equal-weight start
            n_active = len(self._active_universe)
            if n_active > 0:
                w = 1.0 / (n_active + 1)     # +1 for cash
                for p in self._active_universe:
                    s = self._permno_to_slot.get(p)
                    if s is not None and s < self.max_N:
                        self._weights[s] = w
                self._cash = w
        # else: "cash" — already initialised to 100 % cash

        # ---- build observation ----------------------------------------------
        obs, mask = self.interp.interpret(
            first_month,
            self._active_universe,
            self._weights,
            self._cash,
        )
        return {"obs": obs, "action_mask": mask}, {}

    # ================================================================== #
    #  step()
    # ================================================================== #

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """
        Execute one monthly rebalance.

        Sequence
        --------
        1. Identify current month ``t`` and next month ``t+1``.
        2. **Orphan resolution** — sweep missing stocks to cash.
        3. **Action → target weights** via masked softmax.
        4. **Transaction costs** — spread + square-root impact.
        5. **Portfolio return** — weighted stock returns minus costs.
        6. **Reward** = net portfolio return − benchmark return.
        7. Advance weights to reflect realised returns.
        """
        _ep_len   = len(self._episode_months)                      # actual length
        month_t   = int(self._episode_months[self._step_idx])
        is_last   = (self._step_idx >= _ep_len - 2)
        next_step = self._step_idx + 1
        # The return is earned *during* month_t.  At the end of step(),
        # we advance to month t+1 for the next observation.
        month_t1 = int(self._episode_months[min(next_step, len(self._episode_months) - 1)])

        info: Dict[str, Any] = {"month": month_t}

        # ---- 1. orphan resolution (Sweep to Cash) ---------------------------
        sweep_loss = self._sweep_orphans(month_t)

        # ---- 2. compute target weights from action --------------------------
        current_mask = self._build_live_mask(month_t)
        target_weights = masked_softmax(action, current_mask)

        # Separate cash and stock target weights
        target_cash   = float(target_weights[-1])
        target_stocks = target_weights[:-1].copy()

        # ---- 3. transaction costs -------------------------------------------
        delta_w = target_stocks - self._weights          # (max_N,)
        delta_cash = target_cash - self._cash

        total_spread_cost, total_impact_cost = self._calculate_costs(
            month_t, delta_w
        )
        total_cost = total_spread_cost + total_impact_cost

        # ---- 4. apply target weights (after cost deduction) -----------------
        # Costs are deducted proportionally from the portfolio.
        cost_drag = total_cost  # as a fraction of portfolio value
        self._weights = target_stocks.copy()
        self._cash    = target_cash

        # ---- 5. portfolio return for month t --------------------------------
        gross_return = self._compute_gross_return(month_t)
        net_return   = gross_return - cost_drag + sweep_loss  # sweep_loss ≤ 0

        # ---- 6. benchmark & reward ------------------------------------------
        bench_ret = self.interp.get_benchmark_return(month_t)
        if np.isnan(bench_ret):
            bench_ret = self.bench_fallback
        reward = float(net_return - bench_ret)

        # ---- 7. update portfolio value & advance weights --------------------
        self._portfolio_value *= (1.0 + net_return)
        self._advance_weights_for_returns(month_t)

        # ---- 8. advance step ------------------------------------------------
        self._step_idx = next_step
        terminated = (self._step_idx >= _ep_len - 1) 
        truncated  = False

        # ---- 9. next observation --------------------------------------------
        obs, mask = self.interp.interpret(
            month_t1,
            self._active_universe,
            self._weights,
            self._cash,
        )

        # ---- 10. risk-free rate on cash (for logging; already in gross) -----
        rf_earned = self._get_monthly_rf(month_t) * self._cash

        info.update({
            "gross_return":    gross_return,
            "net_return":      net_return,
            "spread_cost":     total_spread_cost,
            "impact_cost":     total_impact_cost,
            "sweep_loss":      sweep_loss,
            "benchmark_ret":   bench_ret,
            "active_return":   reward,
            "portfolio_value": self._portfolio_value,
            "n_active":        len(self._active_universe),
            "cash_weight":     self._cash,
            "turnover":        float(np.abs(delta_w).sum() + abs(delta_cash)),
            "rf_earned":       rf_earned,
        })

        return {"obs": obs, "action_mask": mask}, reward, terminated, truncated, info

    # ================================================================== #
    #  Orphan Resolution: Sweep to Cash  (vectorised)
    # ================================================================== #

    def _sweep_orphans(self, yyyymm: int) -> float:
        """
        Vectorised orphan sweep.  Uses the slot-aligned ``present`` array
        instead of per-slot dict lookups.

        Logic per held slot:
          • present in data          → keep (no action)
          • missing, NOT in active   → dropout artefact → sweep to cash, 0% loss
          • missing, in active       → genuine delist   → apply dlret or −100%
        """
        arrays  = self.interp.get_month_slot_arrays(yyyymm)
        present = arrays["present"]                       # (max_N,) bool

        held = self._weights > 1e-10                      # (max_N,) bool
        missing_held = held & ~present                    # held but missing

        if not np.any(missing_held):
            return 0.0

        loss = 0.0
        for slot in np.where(missing_held)[0]:
            w = self._weights[slot]
            permno = self._slot_to_permno.get(int(slot))
            self._weights[slot] = 0.0

            if permno is None or not self._active_slots_mask[slot]:
                # Dropout artefact or unknown → sweep to cash, no loss
                self._cash += w
            else:
                # Genuine delist
                if self.use_delisting:
                    dlret = self.interp.get_delisting_return(permno, yyyymm)
                    if dlret is not None:
                        recovered = w * (1.0 + dlret)
                        self._cash += max(recovered, 0.0)
                        loss += min(dlret, 0.0) * w
                    else:
                        loss -= w
                else:
                    loss -= w

        return loss

    # ================================================================== #
    #  Transaction Cost Calculation  (vectorised)
    # ================================================================== #

    _SQRT_12 = np.sqrt(12.0)

    def _calculate_costs(
        self, yyyymm: int, delta_w: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fully vectorised spread + square-root impact costs.

        No Python loops over 2,200 slots — pure numpy broadcast.
        """
        arrays = self.interp.get_month_slot_arrays(yyyymm)
        abs_dw = np.abs(delta_w)                          # (max_N,)

        # ---- spread cost = Σ |Δw_i| × spread_i / 2 ----
        total_spread = float(np.dot(abs_dw, arrays["BidAskSpread"]) / 2.0)

        # ---- square-root impact ----
        dolvol_log = arrays["DolVol"]                     # ln($ vol in millions)
        volmkt_ann = arrays["VolMkt"]                     # annualised

        # Safely convert DolVol: exp(log_millions) × 1e6 → raw USD
        safe_dolvol = np.where(dolvol_log > 0, dolvol_log, 30.0)  # 30 → exp(30)*1e6 ≈ huge
        dolvol_usd = np.exp(safe_dolvol) * 1e6

        vol_monthly = volmkt_ann / self._SQRT_12

        capital = self._portfolio_value
        trade_value = abs_dw * capital                    # (max_N,)
        participation = trade_value / dolvol_usd          # (max_N,)

        # impact_i = η × σ_monthly × √participation
        impact = self.impact_eta * vol_monthly * np.sqrt(np.maximum(participation, 0.0))

        # total impact cost = Σ impact_i × |Δw_i|
        total_impact = float(np.dot(impact, abs_dw))

        return total_spread, total_impact

    # ================================================================== #
    #  Risk-Free Rate Helper
    # ================================================================== #

    def _get_monthly_rf(self, yyyymm: int) -> float:
        """
        Return the monthly risk-free rate for the given month.
        If disabled or unavailable, returns 0.0.
        """
        if not self.use_riskfree:
            return 0.0
        rf = self.interp.get_riskfree_rate(yyyymm)
        if self.rf_annualise and rf > 0:
            rf = (1.0 + rf) ** (1.0 / 12.0) - 1.0
        return rf

    # ================================================================== #
    #  Portfolio Return  (vectorised)
    # ================================================================== #

    def _compute_gross_return(self, yyyymm: int) -> float:
        """
        Vectorised: ``dot(weights, returns) + cash × rf``.
        Single numpy dot product replaces 2,200-iteration loop.
        """
        arrays = self.interp.get_month_slot_arrays(yyyymm)
        rf = self._get_monthly_rf(yyyymm)
        stock_ret = float(np.dot(self._weights, arrays["ret"]))
        return stock_ret + self._cash * rf

    # ================================================================== #
    #  Weight Drift After Returns  (vectorised)
    # ================================================================== #

    def _advance_weights_for_returns(self, yyyymm: int) -> None:
        """
        Vectorised weight drift:
            w_new = w × (1 + ret) / (1 + port_ret)
        Single numpy multiply replaces 2,200-iteration loop.
        """
        arrays   = self.interp.get_month_slot_arrays(yyyymm)
        rf       = self._get_monthly_rf(yyyymm)
        port_ret = self._compute_gross_return(yyyymm)
        denom    = 1.0 + port_ret
        if abs(denom) < 1e-12:
            denom = 1.0

        # Vectorised: w_new[i] = w[i] × (1 + ret[i]) / denom
        self._weights = self._weights * (1.0 + arrays["ret"]) / denom
        self._cash    = self._cash * (1.0 + rf) / denom

        # Renormalise for floating-point safety
        total = self._weights.sum() + self._cash
        if total > 1e-12:
            self._weights /= total
            self._cash    /= total

    # ================================================================== #
    #  Action Mask  (vectorised)
    # ================================================================== #

    def _build_live_mask(self, yyyymm: int) -> np.ndarray:
        """
        Vectorised: ``active_slots_mask & present`` in one numpy AND.
        """
        arrays = self.interp.get_month_slot_arrays(yyyymm)
        mask = np.zeros(self.max_N + 1, dtype=np.float32)
        mask[:self.max_N] = (self._active_slots_mask & arrays["present"]).astype(np.float32)
        mask[-1] = 1.0   # cash always active
        return mask