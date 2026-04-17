#!/usr/bin/env python3
"""
train.py
========
Main entry point for training the monthly portfolio RL agent.

Usage
-----
    python train.py                           # defaults from config.yaml
    python train.py --config my_config.yaml   # custom config
    python train.py --algo SAC                # override algorithm
    python train.py --eval-only               # run eval without training

Pipeline Steps
--------------
    1. Load config.yaml
    2. Load pre-processed data (parquet → StateInterpreter)
    3. Build MonthlyPortfolioEnv (train + eval)
    4. Build policy network with SharedWeightExtractor
    5. Train with PPO / TD3 / SAC via Stable-Baselines3
    6. Evaluate and save checkpoints
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

import wandb

from monthly_state_interpreter import MonthlyStateInterpreter
from monthly_portfolio_env import MonthlyPortfolioEnv
from policy_network import make_policy_kwargs, DiscretizedPortfolioWrapper
from wandb_callback import WandbPortfolioCallback, WandbEvalCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("train")

from stable_baselines3.common.callbacks import BaseCallback

# ═══════════════════════════════════════════════════════════════════════════
#  Callback outputting state action values
# ═══════════════════════════════════════════════════════════════════════════


class StateActionLoggerCallback(BaseCallback):
    """
    Logs the raw observation (state) and action every ``log_every`` completed
    episodes.  Works with vectorised envs — tracks per-env episode counts.

    Uses a deep-copy buffer to avoid SB3's ``th.as_tensor`` memory aliasing:
    ``obs_tensor`` may share memory with internal buffers that get overwritten
    when VecEnv auto-resets on ``done=True``.

    Saves an .npz with keys:
        obs          – flat observation vector (features ‖ weights)
        action_mask  – (max_N+1,) active-stock mask
        action       – (max_N+1,) raw policy logits
        feature_cols – 1-D string array of feature column names
    """

    def __init__(
        self,
        log_every: int = 5000,
        log_dir: str = "artifacts/logs",
        feature_cols: list[str] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_every     = log_every
        self.log_dir       = log_dir
        self.feature_cols  = feature_cols or []
        self._ep_counts    = None
        self._total_eps    = 0
        self._obs_buffer   = None     # deep-copied obs, refreshed every step

    def _on_step(self) -> bool:
        if self._ep_counts is None:
            self._ep_counts = np.zeros(self.training_env.num_envs, dtype=int)

        # Deep-copy obs_tensor on every step (~700 KB, negligible).
        # This guarantees we hold the observation the policy acted on,
        # even after VecEnv auto-resets overwrite the underlying buffer.
        if "obs_tensor" in self.locals:
            self._obs_buffer = {
                k: v.cpu().numpy().copy()
                for k, v in self.locals["obs_tensor"].items()
            }

        dones = self.locals["dones"]

        for env_idx, done in enumerate(dones):
            if not done:
                continue

            self._ep_counts[env_idx] += 1
            self._total_eps += 1

            if self._total_eps % self.log_every != 0:
                continue

            if self._obs_buffer is None:
                logger.warning(f"[StateActionLogger] ep {self._total_eps}: "
                               f"no obs_buffer — skipping.")
                continue

            obs_np    = {k: v[env_idx] for k, v in self._obs_buffer.items()}
            action_np = self.locals["actions"][env_idx]
            mask      = obs_np["action_mask"]
            n_active  = int(mask[:-1].sum())

            # ── Save to disk ─────────────────────────────────────────
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(self.log_dir) / f"state_action_ep{self._total_eps}.npz"

            np.savez(
                out_path,
                obs=obs_np["obs"],
                action_mask=mask,
                action=action_np,
                feature_cols=np.array(self.feature_cols, dtype=str),
            )

            # ── W&B logging ──────────────────────────────────────────
            if n_active > 0:
                active_logits = action_np[:-1][mask[:-1] > 0.5]
                wandb.log({
                    "debug/action_logits_hist": wandb.Histogram(active_logits),
                    "debug/n_active_stocks":    n_active,
                    "debug/max_logit":          float(active_logits.max()),
                    "debug/cash_logit":         float(action_np[-1]),
                    "debug/episode":            self._total_eps,
                }, step=self.num_timesteps, commit=False)

            logger.info(
                f"[StateActionLogger] ep {self._total_eps} → {out_path}  "
                f"(n_active={n_active})"
            )

        return True


# ═══════════════════════════════════════════════════════════════════════════
#  Algo Registry
# ═══════════════════════════════════════════════════════════════════════════

ALGO_REGISTRY = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading (from Parquet — Qlib optional)
# ═══════════════════════════════════════════════════════════════════════════

def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame,
                                   Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load all data sources.

    Designed to consume files produced by the ranking pipeline's DataManager:
        data/dataset.parquet      ← DataManager.get_data()
        data/ret_sp500.parquet    ← DataManager.get_ret_sp500()
        data/rf.parquet           ← DataManager.get_rf()

    Also loads optional files produced by helper scripts:
        data/delisting_returns.parquet  ← prepare_delisting_returns.py
        data/xgboost_rank_oos.parquet   ← export_alpha_signal.py

    Returns (data_df, sp500_df, riskfree_df, delisting_df).
    """
    raw_path   = Path(cfg["paths"]["raw_parquet"])
    sp500_path = Path(cfg["paths"]["sp500_parquet"])

    # ---- 1. Main dataset (from DataManager.get_data()) ----
    logger.info(f"Loading main data: {raw_path}")
    df = pd.read_parquet(raw_path)
    df["yyyymm"] = df["yyyymm"].astype(int)
    df["permno"] = df["permno"].astype(int)

    # Match DataManager._preprocess_features: replace ±inf with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    logger.info(f"  {len(df):,} rows, {df['permno'].nunique()} stocks, "
                f"{df['yyyymm'].nunique()} months, "
                f"{len(numeric_cols)} numeric columns")

    # ---- 2. SP500 benchmark (from DataManager.get_ret_sp500()) ----
    #   DataManager stores yyyymm as string ('%Y%m'); we normalise to int.
    logger.info(f"Loading benchmark: {sp500_path}")
    sp500 = pd.read_parquet(sp500_path)
    sp500["yyyymm"] = sp500["yyyymm"].astype(str).str.strip().astype(int)
    sp500["ret"]    = sp500["ret"].astype(float)
    logger.info(f"  {len(sp500)} months "
                f"({sp500['yyyymm'].min()}–{sp500['yyyymm'].max()})")

    # ---- 3. Risk-free rate (from DataManager.get_rf()) ----
    #   DataManager pulls from ff.factors_monthly and divides by 100,
    #   so rf is already a monthly decimal fraction (e.g. 0.003 = 0.3%).
    #   WRDS returns yyyymm as float64 — cast to int.
    rf_path = Path(cfg["paths"].get("riskfree_parquet", ""))
    riskfree_df: Optional[pd.DataFrame] = None
    if rf_path.name and rf_path.exists():
        logger.info(f"Loading risk-free rates: {rf_path}")
        riskfree_df = pd.read_parquet(rf_path)
        riskfree_df["yyyymm"] = riskfree_df["yyyymm"].astype(float).astype(int)
        riskfree_df["rf"]     = riskfree_df["rf"].astype(float)
        logger.info(f"  {len(riskfree_df)} months of RF data "
                    f"({riskfree_df['yyyymm'].min()}–{riskfree_df['yyyymm'].max()})")
    else:
        logger.warning(f"Risk-free rate file not found at {rf_path} — cash earns 0%")

    # ---- 4. Delisting returns (from prepare_delisting_returns.py) ----
    dl_path = Path(cfg["paths"].get("delisting_parquet", ""))
    delisting_df: Optional[pd.DataFrame] = None
    if dl_path.name and dl_path.exists():
        logger.info(f"Loading delisting returns: {dl_path}")
        delisting_df = pd.read_parquet(dl_path)
        delisting_df["permno"] = delisting_df["permno"].astype(int)
        delisting_df["yyyymm"] = delisting_df["yyyymm"].astype(int)
        logger.info(f"  {len(delisting_df)} delisting events loaded")
    else:
        logger.warning(f"Delisting file not found at {dl_path} — using −100% assumption")

    # ---- 5. Alpha signal (from export_alpha_signal.py) ----
    alpha_path = Path(cfg["alpha_signal"]["path"])
    alpha_col  = cfg["alpha_signal"]["col_name"]
    if alpha_path.exists():
        logger.info(f"Merging alpha signal: {alpha_path}")
        alpha_df = pd.read_parquet(alpha_path)
        alpha_df["yyyymm"] = alpha_df["yyyymm"].astype(int)
        alpha_df["permno"] = alpha_df["permno"].astype(int)
        df = df.merge(alpha_df[["permno", "yyyymm", alpha_col]],
                      on=["permno", "yyyymm"], how="left")
        coverage = df[alpha_col].notna().mean()
        logger.info(f"  Alpha coverage: {coverage:.1%} of stock-months")
    else:
        logger.info(f"Alpha signal not found at {alpha_path} — training with 3 base features")

    return df, sp500, riskfree_df, delisting_df


# ═══════════════════════════════════════════════════════════════════════════
#  Environment Factory
# ═══════════════════════════════════════════════════════════════════════════

def make_env(
    interpreter: MonthlyStateInterpreter,
    cfg: dict,
    mode: str = "train",
    seed: int = 0,
) -> gym.Env:
    """Instantiate, Monitor-wrap, and seed a MonthlyPortfolioEnv."""
    env = MonthlyPortfolioEnv(interpreter, cfg, mode=mode)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_vec_env(
    interpreter: MonthlyStateInterpreter,
    cfg: dict,
    n_envs: int = 4,
    mode: str = "train",
    seed: int = 42,
) -> DummyVecEnv:
    """Create a vectorised env with ``n_envs`` parallel instances."""
    def _factory(i: int):
        def _thunk():
            return make_env(interpreter, cfg, mode=mode, seed=seed + i)
        return _thunk

    return DummyVecEnv([_factory(i) for i in range(n_envs)])


# ═══════════════════════════════════════════════════════════════════════════
#  Feature Column Resolution
# ═══════════════════════════════════════════════════════════════════════════

def resolve_feature_cols(cfg: dict, data_df: pd.DataFrame) -> list[str]:
    """
    Determine which feature columns to use in the state.
    Returns optional signals (if present) + base features.
    """
    base = list(cfg["features"]["base_features"])
    prefix: list[str] = []

    for key in ["optional_alpha", "optional_meta"]:
        col = cfg["features"].get(key)
        if col and col in data_df.columns:
            coverage = data_df[col].notna().mean()
            logger.info(f"Signal '{col}' found — coverage {coverage:.1%}")
            prefix.append(col)
        elif col:
            logger.warning(f"Signal '{col}' not in data — skipping.")

    result = prefix + base
    logger.info(f"State features ({len(result)}): {result}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train RL portfolio agent")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--algo",      default=None,
                        help="Override algo: PPO, SAC, TD3")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, run evaluation only")
    parser.add_argument("--load-model", default=None,
                        help="Path to a saved model .zip to resume from")
    args = parser.parse_args()

    # ---- 1. load config ---
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    algo_name  = args.algo or cfg["policy"]["algo"]
    algo_cls   = ALGO_REGISTRY.get(algo_name.upper())
    if algo_cls is None:
        logger.error(f"Unknown algo '{algo_name}'. Choose from: {list(ALGO_REGISTRY)}")
        sys.exit(1)

    train_cfg  = cfg["training"]
    policy_cfg = cfg["policy"]
    seed       = int(train_cfg["seed"])

    # ---- 2. load data ---
    data_df, sp500_df, riskfree_df, delisting_df = load_data(cfg)

    # ---- 3. resolve features ---
    feature_cols = resolve_feature_cols(cfg, data_df)
    n_features   = len(feature_cols)
    max_N        = int(cfg["env"]["max_universe_size"])

    logger.info(f"State features ({n_features}): {feature_cols}")

    # ---- 4. build state interpreter ---
    interpreter = MonthlyStateInterpreter(
        data_df=data_df,
        sp500_df=sp500_df,
        feature_cols=feature_cols,
        return_col=cfg["features"]["return_col"],
        mcap_col=cfg["features"]["mcap_col"],
        max_universe_size=max_N,
        riskfree_df=riskfree_df,
        delisting_df=delisting_df,
    )

    # ---- 5. build envs ---
    n_envs   = int(train_cfg["n_envs"])
    train_env = make_vec_env(interpreter, cfg, n_envs=n_envs, mode="train", seed=seed)
    eval_env  = make_vec_env(interpreter, cfg, n_envs=1,      mode="val",   seed=seed + 1000)

    # ---- 6. build policy ---
    policy_kwargs = make_policy_kwargs(
        n_features=n_features,
        max_universe_size=max_N,
        encoder_hidden_dims=policy_cfg["stock_encoder"]["hidden_dims"],
        actor_hidden_dims=policy_cfg["actor"]["hidden_dims"],
        critic_hidden_dims=policy_cfg["critic"]["hidden_dims"],
        activation=policy_cfg["stock_encoder"]["activation"],
    )

    # ---- 7. initialise Weights & Biases ---
    wandb_cfg = cfg.get("wandb", {})
    run_name = wandb_cfg.get("run_name") or f"{algo_name}_{seed}"
    wandb.init(
        project=wandb_cfg.get("project", "rl-portfolio"),
        name=run_name,
        config=cfg,
        tags=wandb_cfg.get("tags", [algo_name, f"features_{n_features}"]),
        save_code=True,
    )

    # ---- 8. create or load model ---
    model_dir = Path(cfg["paths"]["model_dir"]).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best").mkdir(parents=True, exist_ok=True)
    (model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ---- Device selection (auto / cuda / cpu) ----
    device = train_cfg.get("device", "auto")
    logger.info(f"Device: {device}")

    common_kwargs = dict(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=float(train_cfg["learning_rate"]),
        seed=seed,
        verbose=1,
        device=device,
        tensorboard_log=None,       # W&B replaces TensorBoard
        policy_kwargs=policy_kwargs,
    )

    # Algo-specific kwargs
    if algo_name.upper() == "PPO":
        common_kwargs.update(
            n_steps=int(train_cfg.get("n_steps", 2048)),
            batch_size=int(train_cfg["batch_size"]),
            n_epochs=int(train_cfg["n_epochs"]),
            gamma=float(train_cfg["gamma"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
            clip_range=float(train_cfg["clip_range"]),
            target_kl=train_cfg.get("target_kl", None),  # early-stop PPO epochs
            ent_coef=float(train_cfg["ent_coef"]),
            vf_coef=float(train_cfg["vf_coef"]),
            max_grad_norm=float(train_cfg["max_grad_norm"]),
        )
    elif algo_name.upper() in ("SAC", "TD3"):
        common_kwargs.update(
            batch_size=int(train_cfg["batch_size"]),
            gamma=float(train_cfg["gamma"]),
        )
        # Remove PPO-only keys
        for k in ["n_epochs", "gae_lambda", "clip_range", "ent_coef", "vf_coef"]:
            common_kwargs.pop(k, None)

    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        model = algo_cls.load(args.load_model, env=train_env)
    else:
        model = algo_cls(**common_kwargs)

    logger.info(f"Model architecture:\n{model.policy}")
    wandb.config.update({"model/param_count": sum(
        p.numel() for p in model.policy.parameters()
    )}, allow_val_change=True)

    # ---- 9. callbacks ---
    eval_freq = int(train_cfg["eval_freq"]) // n_envs

    # W&B portfolio callback — logs every step + episode summaries
    wandb_portfolio_cb = WandbPortfolioCallback(
        log_freq=1,
        episode_log=True,
        save_freq=int(train_cfg["save_freq"]),
        model_save_path=str(model_dir / "wandb_checkpoints"),
        verbose=1,
    )

    # SB3 eval callback — still runs periodic deterministic evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=int(train_cfg["n_eval_episodes"]),
        deterministic=True,
    )

    # W&B eval callback — captures SB3 internal metrics (loss, entropy, etc.)
    wandb_eval_cb = WandbEvalCallback(
        eval_freq=eval_freq * n_envs,
    )

    # ---- 10. train ---
    if not args.eval_only:
        total_steps = int(train_cfg["total_timesteps"])
        logger.info(f"Starting training: {algo_name}, {total_steps:,} steps, "
                     f"{n_envs} parallel envs")
        state_action_cb = StateActionLoggerCallback(
            log_every=5000,
            log_dir=cfg["paths"]["log_dir"],
            feature_cols=feature_cols,
            verbose=1,
        )
        model.learn(
            total_timesteps=total_steps,
            callback=[wandb_portfolio_cb, eval_callback, wandb_eval_cb, state_action_cb],  # ← added
            # callback=[wandb_portfolio_cb, eval_callback, wandb_eval_cb],
            progress_bar=True,
        )
        final_path = model_dir / "final_model"
        model.save(str(final_path))
        logger.info(f"Final model saved to {final_path}")
    else:
        logger.info("Eval-only mode — skipping training.")

    # ---- 11. final test evaluation → W&B ---
    logger.info("Running final evaluation on test set …")
    eval_single = make_env(interpreter, cfg, mode="test", seed=seed + 9999)
    obs, _ = eval_single.reset()
    total_reward = 0.0
    done = False
    step_log = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_single.step(action)
        done = terminated or truncated
        total_reward += reward
        step_log.append(info)

    # Compute test-set summary metrics
    test_active  = [s.get("active_return", 0) for s in step_log]
    test_net     = [s.get("net_return", 0)    for s in step_log]
    test_costs   = [s.get("spread_cost", 0) + s.get("impact_cost", 0) for s in step_log]
    test_turn    = [s.get("turnover", 0)      for s in step_log]
    test_pv      = [s.get("portfolio_value", 1) for s in step_log]

    pv_arr = np.array(test_pv)
    peak   = np.maximum.accumulate(pv_arr)
    dd     = (peak - pv_arr) / np.where(peak > 0, peak, 1.0)

    test_sharpe = 0.0
    if len(test_active) > 1 and np.std(test_active) > 1e-10:
        test_sharpe = (np.mean(test_active) / np.std(test_active)) * np.sqrt(12)

    wandb.log({
        "test/cumulative_active_return": float(np.sum(test_active)),
        "test/cumulative_net_return":    float(np.sum(test_net)),
        "test/sharpe_ratio":             test_sharpe,
        "test/max_drawdown":             float(np.max(dd)),
        "test/mean_turnover":            float(np.mean(test_turn)),
        "test/total_cost":               float(np.sum(test_costs)),
        "test/final_portfolio_value":    float(test_pv[-1]) if test_pv else 0.0,
        "test/n_months":                 len(step_log),
    })

    # Log test equity curve as a W&B table
    test_table = wandb.Table(
        columns=["month", "active_return", "net_return", "portfolio_value",
                 "turnover", "cash_weight", "spread_cost", "impact_cost"],
        data=[
            [
                s.get("month", i),
                s.get("active_return", 0),
                s.get("net_return", 0),
                s.get("portfolio_value", 0),
                s.get("turnover", 0),
                s.get("cash_weight", 0),
                s.get("spread_cost", 0),
                s.get("impact_cost", 0),
            ]
            for i, s in enumerate(step_log)
        ],
    )
    wandb.log({"test/equity_curve": test_table})

    logger.info(f"Test episode — cumulative active return: {total_reward:.4f}, "
                f"Sharpe: {test_sharpe:.2f}")

    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()