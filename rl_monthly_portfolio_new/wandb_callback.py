"""
wandb_callback.py
==================
Custom Stable-Baselines3 callback that logs the rich ``info`` dict from
MonthlyPortfolioEnv to Weights & Biases.

Logged Metrics
--------------
**Per-step** (every env step, under ``step/``):
    active_return, net_return, gross_return, spread_cost, impact_cost,
    sweep_loss, benchmark_ret, portfolio_value, cash_weight, turnover,
    n_active, rf_earned

**Per-episode** (at episode end, under ``episode/``):
    cumulative_active_return, cumulative_net_return, mean_turnover,
    max_drawdown, sharpe_ratio, n_steps, final_portfolio_value

**Evaluation** (from EvalCallback, under ``eval/``):
    mean_reward, std_reward (patched in via a companion eval callback)

**System**:
    learning_rate, clip_range, entropy_loss, policy_loss, value_loss
    (captured from SB3's internal logger)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class WandbPortfolioCallback(BaseCallback):
    """
    SB3 callback that streams portfolio metrics to W&B.

    Parameters
    ----------
    log_freq : int
        Log step-level metrics every ``log_freq`` environment steps.
        Set to 1 for full granularity (default), higher to reduce volume.
    episode_log : bool
        If True, compute and log episode-level summary statistics
        (Sharpe, drawdown, cumulative return) at every episode boundary.
    save_freq : int
        Save model checkpoint as a W&B artifact every ``save_freq`` steps.
        Set to 0 to disable artifact saving.
    model_save_path : str
        Directory for temporary model saves before uploading as artifacts.
    verbose : int
        Verbosity level (0 = silent, 1 = info, 2 = debug).
    """

    def __init__(
        self,
        log_freq: int = 1,
        episode_log: bool = True,
        save_freq: int = 100_000,
        model_save_path: str = "artifacts/models/wandb_checkpoints",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_log = episode_log
        self.save_freq = save_freq
        self.model_save_path = model_save_path

        # ---- per-episode accumulators (one per vec-env slot) ----
        self._episode_buffers: Dict[int, list[Dict[str, float]]] = defaultdict(list)
        self._episode_count = 0
        self._start_time = 0.0

    # ================================================================== #
    #  SB3 Callback Interface
    # ================================================================== #

    def _on_training_start(self) -> None:
        """Called once before the first rollout."""
        self._start_time = time.time()

        # Log SB3 hyperparameters that aren't in our config.yaml
        wandb.config.update({
            "sb3/n_envs":        self.training_env.num_envs,
            "sb3/policy_class":  type(self.model.policy).__name__,
            "sb3/device":        str(self.model.device),
        }, allow_val_change=True)

    def _on_step(self) -> bool:
        """Called after every env.step() across all vec-envs."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_idx, info in enumerate(infos):
            if not info:
                continue

            # ---- accumulate for episode summary ----
            if self.episode_log:
                self._episode_buffers[env_idx].append(info)

            # ---- step-level logging ----
            if self.num_timesteps % self.log_freq == 0:
                step_metrics = {}
                for key in [
                    "active_return", "net_return", "gross_return",
                    "spread_cost", "impact_cost", "sweep_loss",
                    "benchmark_ret", "portfolio_value", "cash_weight",
                    "turnover", "n_active", "rf_earned",
                ]:
                    val = info.get(key)
                    if val is not None:
                        step_metrics[f"step/{key}"] = float(val)

                if step_metrics:
                    step_metrics["step/env_step"] = self.num_timesteps
                    wandb.log(step_metrics, step=self.num_timesteps)

            # ---- episode boundary ----
            if dones[env_idx] and self.episode_log:
                self._log_episode_summary(env_idx)

        # ---- periodic model checkpoint as W&B artifact ----
        if self.save_freq > 0 and self.num_timesteps % self.save_freq == 0:
            self._save_model_artifact()

        return True  # returning False would stop training

    def _on_training_end(self) -> None:
        """Called once after the last rollout."""
        elapsed = time.time() - self._start_time
        wandb.log({
            "train/total_episodes":   self._episode_count,
            "train/total_timesteps":  self.num_timesteps,
            "train/wall_time_sec":    elapsed,
            "train/steps_per_sec":    self.num_timesteps / max(elapsed, 1),
        }, step=self.num_timesteps)

        # Final model save
        if self.save_freq > 0:
            self._save_model_artifact(tag="final")

    # ================================================================== #
    #  Episode Summary
    # ================================================================== #

    def _log_episode_summary(self, env_idx: int) -> None:
        """Compute and log episode-level statistics."""
        buf = self._episode_buffers.pop(env_idx, [])
        if not buf:
            return

        self._episode_count += 1

        # ---- extract time series from info dicts ----
        active_returns = np.array([s.get("active_return", 0.0) for s in buf])
        net_returns    = np.array([s.get("net_return", 0.0)    for s in buf])
        turnovers      = np.array([s.get("turnover", 0.0)      for s in buf])
        spread_costs   = np.array([s.get("spread_cost", 0.0)   for s in buf])
        impact_costs   = np.array([s.get("impact_cost", 0.0)   for s in buf])
        port_values    = np.array([s.get("portfolio_value", 1.0) for s in buf])

        n_steps = len(buf)

        # ---- cumulative returns ----
        cum_active = float(np.sum(active_returns))
        cum_net    = float(np.sum(net_returns))

        # ---- Sharpe ratio (monthly, annualised) ----
        if n_steps > 1 and np.std(active_returns) > 1e-10:
            sharpe = (np.mean(active_returns) / np.std(active_returns)) * np.sqrt(12)
        else:
            sharpe = 0.0

        # ---- max drawdown (on portfolio value curve) ----
        peak = np.maximum.accumulate(port_values)
        drawdown = (peak - port_values) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # ---- log ----
        episode_metrics = {
            "episode/cumulative_active_return": cum_active,
            "episode/cumulative_net_return":    cum_net,
            "episode/sharpe_ratio":             sharpe,
            "episode/max_drawdown":             max_dd,
            "episode/mean_turnover":            float(np.mean(turnovers)),
            "episode/total_spread_cost":        float(np.sum(spread_costs)),
            "episode/total_impact_cost":        float(np.sum(impact_costs)),
            "episode/total_cost":               float(np.sum(spread_costs) + np.sum(impact_costs)),
            "episode/n_steps":                  n_steps,
            "episode/final_portfolio_value":    float(port_values[-1]) if len(port_values) > 0 else 0.0,
            "episode/count":                    self._episode_count,
        }
        wandb.log(episode_metrics, step=self.num_timesteps)

        if self.verbose >= 1:
            print(f"  [Episode {self._episode_count}]  "
                  f"active_ret={cum_active:+.4f}  "
                  f"sharpe={sharpe:.2f}  "
                  f"max_dd={max_dd:.2%}  "
                  f"turnover={np.mean(turnovers):.3f}")

    # ================================================================== #
    #  Model Artifact Saving
    # ================================================================== #

    def _save_model_artifact(self, tag: str = "checkpoint") -> None:
        """Save current model as a W&B artifact."""
        from pathlib import Path
        save_dir = Path(self.model_save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"rl_portfolio_{self.num_timesteps}"
        save_path = save_dir / filename
        self.model.save(str(save_path))

        artifact = wandb.Artifact(
            name=f"rl-portfolio-model-{tag}",
            type="model",
            metadata={
                "timesteps": self.num_timesteps,
                "episodes":  self._episode_count,
                "tag":       tag,
            },
        )
        artifact.add_file(str(save_path) + ".zip")
        wandb.log_artifact(artifact)

        if self.verbose >= 1:
            print(f"  [W&B] Saved model artifact ({tag}) at step {self.num_timesteps}")


# ═══════════════════════════════════════════════════════════════════════════
#  Eval Callback with W&B Logging
# ═══════════════════════════════════════════════════════════════════════════

class WandbEvalCallback(BaseCallback):
    """
    Companion callback that logs SB3's EvalCallback results to W&B.

    Attach this as a second callback alongside the standard EvalCallback.
    It reads the eval results that EvalCallback writes to ``self.parent``
    and forwards them to W&B.

    Alternatively, you can subclass EvalCallback directly — but this
    approach is cleaner and doesn't require monkey-patching SB3 internals.

    Usage
    -----
        eval_cb = EvalCallback(eval_env, ...)
        wandb_eval_cb = WandbEvalCallback()
        model.learn(callback=[portfolio_cb, eval_cb, wandb_eval_cb])
    """

    def __init__(self, eval_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self._last_logged_step = 0

    def _on_step(self) -> bool:
        # Check if EvalCallback just ran by looking for its results
        # in the locals/parent. EvalCallback stores results in self.evaluations_results
        # but the simplest approach is to check the logger for eval/ keys.
        if self.num_timesteps - self._last_logged_step >= self.eval_freq:
            # SB3 logs eval metrics to its own logger; we capture them here.
            # The eval metrics are typically in self.logger.name_to_value
            if hasattr(self.model, "logger") and self.model.logger is not None:
                logger_dict = getattr(self.model.logger, "name_to_value", {})
                eval_metrics = {}
                for key, val in logger_dict.items():
                    if "eval" in key.lower() or "rollout" in key.lower():
                        clean_key = key.replace("/", "_").replace("\\", "_")
                        eval_metrics[f"eval/{clean_key}"] = val
                    # Also capture SB3 training internals
                    if any(k in key.lower() for k in
                           ["entropy", "policy_loss", "value_loss",
                            "learning_rate", "clip_range", "explained_variance"]):
                        eval_metrics[f"train/{key}"] = val

                if eval_metrics:
                    wandb.log(eval_metrics, step=self.num_timesteps)
                    self._last_logged_step = self.num_timesteps

        return True
