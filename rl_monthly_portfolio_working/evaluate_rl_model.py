#!/usr/bin/env python3
"""
evaluate_rl_model.py
====================
Inference + P&L evaluation pipeline for the trained RL PPO portfolio model.

Loads the best PPO checkpoint, steps through the test period month-by-month,
extracts portfolio weights, and computes performance metrics matching the
pnl_custom_strategy output format.

Usage
-----
    python evaluate_rl_model.py                            # defaults
    python evaluate_rl_model.py --config config.yaml       # custom config
    python evaluate_rl_model.py --model-path path/to/best  # custom model path
    python evaluate_rl_model.py --bps 10                   # transaction cost in bps
    python evaluate_rl_model.py --test-start 202103 --test-end 202412
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from stable_baselines3 import PPO

from monthly_state_interpreter import MonthlyStateInterpreter
from monthly_portfolio_env import MonthlyPortfolioEnv, masked_softmax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("evaluate")


# ═══════════════════════════════════════════════════════════════════════════
#  1. Data Loading  (mirrors train.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_data(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame,
                                   Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load all data sources from parquet files."""
    raw_path   = Path(cfg["paths"]["raw_parquet"])
    sp500_path = Path(cfg["paths"]["sp500_parquet"])

    # ---- Main dataset ----
    logger.info(f"Loading main data: {raw_path}")
    df = pd.read_parquet(raw_path)
    df["yyyymm"] = df["yyyymm"].astype(int)
    df["permno"] = df["permno"].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    logger.info(f"  {len(df):,} rows, {df['permno'].nunique()} stocks, "
                f"{df['yyyymm'].nunique()} months")

    # ---- SP500 benchmark ----
    logger.info(f"Loading benchmark: {sp500_path}")
    sp500 = pd.read_parquet(sp500_path)
    sp500["yyyymm"] = sp500["yyyymm"].astype(str).str.strip().astype(int)
    sp500["ret"]    = sp500["ret"].astype(float)

    # ---- Risk-free rate ----
    rf_path = Path(cfg["paths"].get("riskfree_parquet", ""))
    riskfree_df = None
    if rf_path.name and rf_path.exists():
        logger.info(f"Loading risk-free rates: {rf_path}")
        riskfree_df = pd.read_parquet(rf_path)
        riskfree_df["yyyymm"] = riskfree_df["yyyymm"].astype(float).astype(int)
        riskfree_df["rf"]     = riskfree_df["rf"].astype(float)

    # ---- Delisting returns ----
    dl_path = Path(cfg["paths"].get("delisting_parquet", ""))
    delisting_df = None
    if dl_path.name and dl_path.exists():
        logger.info(f"Loading delisting returns: {dl_path}")
        delisting_df = pd.read_parquet(dl_path)
        delisting_df["permno"] = delisting_df["permno"].astype(int)
        delisting_df["yyyymm"] = delisting_df["yyyymm"].astype(int)

    # ---- Alpha signal (optional) ----
    alpha_path = Path(cfg["alpha_signal"]["path"])
    alpha_col  = cfg["alpha_signal"]["col_name"]
    if alpha_path.exists():
        logger.info(f"Merging alpha signal: {alpha_path}")
        alpha_df = pd.read_parquet(alpha_path)
        alpha_df["yyyymm"] = alpha_df["yyyymm"].astype(int)
        alpha_df["permno"] = alpha_df["permno"].astype(int)
        df = df.merge(alpha_df[["permno", "yyyymm", alpha_col]],
                      on=["permno", "yyyymm"], how="left")

    return df, sp500, riskfree_df, delisting_df


# ═══════════════════════════════════════════════════════════════════════════
#  2. Feature Resolution  (mirrors train.py)
# ═══════════════════════════════════════════════════════════════════════════

def resolve_feature_cols(cfg: dict, data_df: pd.DataFrame) -> list[str]:
    """Determine which feature columns to use in the state."""
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
#  3. RL Model Inference → Strategy DataFrame
# ═══════════════════════════════════════════════════════════════════════════

def run_rl_inference(
    model: PPO,
    interpreter: MonthlyStateInterpreter,
    cfg: dict,
    test_start: int,
    test_end: int,
) -> pd.DataFrame:
    """
    Run the trained RL model through the test period and extract
    portfolio weights at each month.

    Returns
    -------
    strategy_df : pd.DataFrame
        Columns: [yyyymm, permno, weight]
        permno = -1 represents the risk-free (cash) position.
        Weights sum to 1.0 each month.
    """
    # ---- Override config for full test period ----
    test_cfg = cfg.copy()
    test_cfg = {**cfg}  # shallow copy

    # Deep-copy dates section to override test range
    test_cfg["dates"] = dict(cfg["dates"])
    test_cfg["dates"]["test_start"] = f"{str(test_start)[:4]}-{str(test_start)[4:]}"
    test_cfg["dates"]["test_end"]   = f"{str(test_end)[:4]}-{str(test_end)[4:]}"

    # Compute how many months in the test window
    avail_months = interpreter.get_available_months(test_start, test_end)
    n_months = len(avail_months)
    logger.info(f"Test period: {test_start}–{test_end} → {n_months} months available")

    if n_months == 0:
        logger.error("No months available in the specified test range!")
        return pd.DataFrame(columns=["yyyymm", "permno", "weight"])

    # Override episode_length to cover the full test period
    test_cfg["env"] = dict(cfg["env"])
    test_cfg["env"]["episode_length"] = n_months
    # No dropout during evaluation
    test_cfg["env"]["dropout_rate"] = 0.0

    # ---- Create test environment ----
    env = MonthlyPortfolioEnv(interpreter, test_cfg, mode="test")

    # ---- Run inference ----
    obs, info = env.reset(seed=9999)
    all_records: list[dict] = []
    step_infos: list[dict] = []

    done = False
    step = 0

    while not done:
        # Current month
        month_t = int(env._episode_months[env._step_idx])

        # Get model action
        action, _ = model.predict(obs, deterministic=True)

        # Before stepping, extract TARGET weights that will be applied
        # (Replicate env's masked_softmax logic to capture the weights)
        current_mask = env._build_live_mask(month_t)
        target_weights = masked_softmax(action, current_mask)

        target_cash   = float(target_weights[-1])
        target_stocks = target_weights[:-1]  # (max_N,)

        # Record non-zero stock weights
        nonzero_slots = np.where(target_stocks > 1e-8)[0]
        for slot in nonzero_slots:
            permno = env._slot_to_permno.get(int(slot))
            if permno is not None:
                all_records.append({
                    "yyyymm": month_t,
                    "permno": int(permno),
                    "weight": float(target_stocks[slot]),
                })

        # Record cash position (permno = -1)
        if target_cash > 1e-8:
            all_records.append({
                "yyyymm": month_t,
                "permno": -1,
                "weight": target_cash,
            })

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_infos.append({
            "month": month_t,
            "reward": reward,
            **{k: v for k, v in info.items()
               if isinstance(v, (int, float, np.floating, np.integer))},
        })
        step += 1

    logger.info(f"Inference complete: {step} months, "
                f"{len(all_records)} weight records")

    strategy_df = pd.DataFrame(all_records)

    # ---- Also save step-level info for diagnostics ----
    info_df = pd.DataFrame(step_infos)
    info_path = "rl_step_info.csv"
    info_df.to_csv(info_path, index=False)
    logger.info(f"Step-level info saved to {info_path}")

    return strategy_df


# ═══════════════════════════════════════════════════════════════════════════
#  4. P&L Evaluation  (standalone version of pnl_custom_strategy)
# ═══════════════════════════════════════════════════════════════════════════

def pnl_custom_strategy(
    strategy_df: pd.DataFrame,
    X_test: pd.DataFrame,
    rf_df: pd.DataFrame,
    ret_sp500: pd.DataFrame,
    strategy_name: str = "RL PPO Strategy",
    bps: float = 10,
) -> Dict:
    """
    Portfolio P&L with stocks + risk-free asset + transaction costs.

    Standalone version of the class method — takes data as arguments.

    Parameters
    ----------
    strategy_df : DataFrame with [yyyymm, permno, weight]
    X_test      : DataFrame with [yyyymm, permno, ret_1m, ...]
    rf_df       : DataFrame with [yyyymm, rf]
    ret_sp500   : DataFrame with [yyyymm, ret]
    strategy_name : label for the strategy
    bps         : transaction cost in basis points

    Returns
    -------
    dict with keys: metrics_df, sp_500_ols_metrics, wealth_df, drawdown_df
    """
    strat = strategy_df.copy()
    cost_rate = bps / 10000

    # --- 1. Merge stock returns ---
    data = strat.merge(
        X_test[["yyyymm", "permno", "ret_1m"]].drop_duplicates(),
        on=["yyyymm", "permno"],
        how="left",
    )

    # Prepare shifted rf and sp500 (next-month forward returns)
    rf = rf_df[["yyyymm", "rf"]].copy()
    sp = ret_sp500[["yyyymm", "ret"]].copy()

    rf = rf.sort_values("yyyymm")
    sp = sp.sort_values("yyyymm")
    rf["rf_1m"]       = rf["rf"].shift(-1)
    sp["ret_1m_sp500"] = sp["ret"].shift(-1)

    # --- 2. Merge rf ---
    data = data.merge(rf[["yyyymm", "rf_1m"]], on="yyyymm", how="left")

    # Fill edge cases: if rf is missing for final months, use last known
    if data["rf_1m"].isna().any():
        missing_rf = data[data["rf_1m"].isna()]["yyyymm"].unique()
        logger.warning(f"Missing rf for months: {missing_rf} — forward-filling")
        data["rf_1m"] = data["rf_1m"].ffill().bfill()

    # --- 3. Replace cash return with rf ---
    data["ret_1m"] = np.where(
        data["permno"] == -1,
        data["rf_1m"],
        data["ret_1m"],
    )

    if data["ret_1m"].isna().any():
        n_missing = data["ret_1m"].isna().sum()
        logger.warning(f"{n_missing} missing stock returns — filling with 0")
        data["ret_1m"] = data["ret_1m"].fillna(0.0)

    # --- 4. Gross returns ---
    data["gross_ret"] = data["weight"] * data["ret_1m"]

    # --- 5. Turnover ---
    data = data.sort_values(["permno", "yyyymm"])
    data["prev_weight"] = data.groupby("permno")["weight"].shift(1).fillna(0)
    data["turnover"] = (data["weight"] - data["prev_weight"]).abs()
    data["cost"] = data["turnover"] * cost_rate
    data["net_ret"] = data["gross_ret"] - data["cost"]

    # --- 6. Aggregate portfolio ---
    port = data.groupby("yyyymm", as_index=False)["net_ret"].sum()
    port = port.sort_values("yyyymm")

    # --- 7. Wealth ---
    wealth = 1.0
    wealth_list = []
    for r in port["net_ret"].values:
        wealth *= (1 + r)
        wealth_list.append(wealth)
    port["wealth"] = wealth_list

    df_wealth = port[["yyyymm", "wealth"]].copy()

    # --- 8. Metrics ---
    monthly_ret = port["net_ret"].astype(float)
    n_months = len(monthly_ret)

    mean_ret = monthly_ret.mean() if n_months > 0 else np.nan
    std_ret  = monthly_ret.std(ddof=1) if n_months > 1 else np.nan

    annualized_sharpe = np.nan
    if std_ret and std_ret > 0:
        annualized_sharpe = (mean_ret / std_ret) * np.sqrt(12)

    annualized_return = np.nan
    if n_months > 0:
        ending = df_wealth["wealth"].iloc[-1]
        annualized_return = ending ** (12 / n_months) - 1

    rolling_peak = df_wealth["wealth"].cummax()
    drawdown = df_wealth["wealth"] / rolling_peak - 1
    df_drawdown = pd.DataFrame({
        "yyyymm": df_wealth["yyyymm"],
        "drawdown": drawdown,
    })
    max_drawdown = float(drawdown.min())

    wins   = monthly_ret[monthly_ret > 0]
    losses = monthly_ret[monthly_ret < 0]
    avg_win  = float(wins.mean()) if not wins.empty else np.nan
    avg_loss = float(losses.mean()) if not losses.empty else np.nan

    # --- 9. OLS regression vs SP500 ---
    port["yyyymm"]    = port["yyyymm"].astype(int)
    sp["yyyymm"]      = sp["yyyymm"].astype(int)
    rf["yyyymm"]      = rf["yyyymm"].astype(int)

    merged = port.merge(
        sp[["yyyymm", "ret_1m_sp500"]], on="yyyymm", how="left"
    ).merge(
        rf[["yyyymm", "rf_1m"]], on="yyyymm", how="left"
    )
    merged["excess_ret"]   = (merged["net_ret"] - merged["rf_1m"]).astype(float)
    merged["excess_sp500"] = (merged["ret_1m_sp500"] - merged["rf_1m"]).astype(float)

    # Drop rows where SP500 or rf is missing
    merged_clean = merged.dropna(subset=["excess_ret", "excess_sp500"])

    sp_500_ols_metrics = {}
    if len(merged_clean) > 2:
        X_ols = sm.add_constant(merged_clean["excess_sp500"])
        y_ols = merged_clean["excess_ret"]
        ols_model = sm.OLS(y_ols, X_ols).fit()

        sp_500_ols_metrics = {
            "alpha":         ols_model.params["const"],
            "beta":          ols_model.params["excess_sp500"],
            "r2":            ols_model.rsquared,
            "adj_r2":        ols_model.rsquared_adj,
            "alpha_tstat":   ols_model.tvalues["const"],
            "beta_tstat":    ols_model.tvalues["excess_sp500"],
            "alpha_pvalue":  ols_model.pvalues["const"],
            "beta_pvalue":   ols_model.pvalues["excess_sp500"],
        }

    metrics_df = pd.DataFrame([{
        "Strategy":                  strategy_name,
        "Sharpe Ratio (ann)":        round(annualized_sharpe, 2),
        "Return (% ann)":            round(annualized_return * 100, 1),
        "Max Drawdown (%)":          round(max_drawdown * 100, 1),
        "Avg Win (%)":               round(avg_win * 100, 2),
        "Avg Loss (%)":              round(avg_loss * 100, 2),
        "Total Months":              n_months,
    }])

    return {
        "metrics_df":         metrics_df,
        "sp_500_ols_metrics": sp_500_ols_metrics,
        "wealth_df":          df_wealth,
        "drawdown_df":        df_drawdown,
        "monthly_returns":    port,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  5. Plotting (optional — saves to file)
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(wealth_df: pd.DataFrame, drawdown_df: pd.DataFrame,
                 strategy_name: str, output_dir: Path):
    """Generate and save PnL + drawdown plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Convert yyyymm to datetime for plotting
        wealth_df = wealth_df.copy()
        wealth_df["date"] = pd.to_datetime(
            wealth_df["yyyymm"].astype(str), format="%Y%m"
        )
        drawdown_df = drawdown_df.copy()
        drawdown_df["date"] = pd.to_datetime(
            drawdown_df["yyyymm"].astype(str), format="%Y%m"
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # PnL
        axes[0].plot(wealth_df["date"], wealth_df["wealth"],
                     color="#2563eb", linewidth=2)
        axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Wealth (starting = 1.0)")
        axes[0].set_title(f"{strategy_name} — Cumulative P&L")
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        axes[1].fill_between(drawdown_df["date"],
                             drawdown_df["drawdown"] * 100,
                             color="#ef4444", alpha=0.4)
        axes[1].plot(drawdown_df["date"], drawdown_df["drawdown"] * 100,
                     color="#ef4444", linewidth=1)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_title(f"{strategy_name} — Drawdown")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / f"{strategy_name.replace(' ', '_')}_pnl.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved to {fig_path}")

    except ImportError:
        logger.warning("matplotlib not installed — skipping plots")


# ═══════════════════════════════════════════════════════════════════════════
#  6. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL PPO model and compute P&L metrics"
    )
    parser.add_argument("--config",     default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--model-path", default="artifacts-res/models/best/best_model",
                        help="Path to best model .zip (without .zip extension)")
    parser.add_argument("--test-start", default=202103, type=int,
                        help="Test start month (yyyymm)")
    parser.add_argument("--test-end",   default=202412, type=int,
                        help="Test end month (yyyymm)")
    parser.add_argument("--bps",        default=10, type=float,
                        help="Transaction cost in basis points")
    parser.add_argument("--output-dir", default="artifacts-res",
                        help="Directory for output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load config ----
    logger.info(f"Loading config: {args.config}")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 2. Load data ----
    data_df, sp500_df, riskfree_df, delisting_df = load_data(cfg)

    # ---- 3. Resolve features ----
    feature_cols = resolve_feature_cols(cfg, data_df)
    n_features   = len(feature_cols)
    max_N        = int(cfg["env"]["max_universe_size"])
    logger.info(f"Features: {n_features}, Max universe: {max_N}")

    # ---- 4. Build interpreter ----
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

    # ---- 5. Load model ----
    model_path = args.model_path
    # SB3 .load() appends .zip automatically if missing
    logger.info(f"Loading PPO model from: {model_path}")
    model = PPO.load(model_path, device="cpu")
    logger.info(f"Model loaded. Policy:\n{model.policy}")

    # ---- 6. Run inference ----
    logger.info(f"Running RL inference: {args.test_start} → {args.test_end}")
    strategy_df = run_rl_inference(
        model, interpreter, cfg,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    # Save strategy weights
    weights_path = output_dir / "rl_strategy_weights.csv"
    strategy_df.to_csv(weights_path, index=False)
    logger.info(f"Strategy weights saved to {weights_path}")

    # ---- Weight summary per month ----
    monthly_summary = strategy_df.groupby("yyyymm").agg(
        n_stocks=("permno", lambda x: (x != -1).sum()),
        cash_weight=("weight", lambda x: x[strategy_df.loc[x.index, "permno"] == -1].sum()),
        total_weight=("weight", "sum"),
        top_weight=("weight", "max"),
    )
    logger.info(f"\nMonthly weight summary:\n{monthly_summary.to_string()}")

    # ---- 7. Prepare data for P&L ----
    # X_test: needs at least [yyyymm, permno, ret_1m]
    #
    # TIMING CONVENTION:
    #   In the RL env, weights at month_t earn `ret` at month_t (same month).
    #   In pnl_custom_strategy, weights at yyyymm earn `ret_1m` at yyyymm.
    #
    #   If your X_test has a pre-computed `ret_1m` column (forward return),
    #   and your hand-crafted strategies use weights at t to earn ret from t→t+1,
    #   then the RL weights (which earn ret AT month t) need to be shifted back
    #   by 1 month so that pnl_custom_strategy pairs them correctly.
    #
    #   However, if `ret_1m` in X_test is actually the SAME as `ret` (concurrent
    #   return), then no shift is needed.
    #
    #   Here we use `ret` as `ret_1m` so the P&L is consistent with the RL env's
    #   actual behaviour.  The RL weights at month_t earn returns AT month_t.
    #   This makes the metrics directly comparable to the env's own tracking.
    #
    #   If you want to compare against hand-crafted strategies that use forward
    #   returns, set USE_FORWARD_RETURN = True below.

    USE_FORWARD_RETURN = False  # ← Set True if your hand-crafted strategies use ret_1m as forward return

    test_months = strategy_df["yyyymm"].unique()

    if USE_FORWARD_RETURN and "ret_1m" in data_df.columns:
        # Use the pre-computed forward return from your dataset
        X_test = data_df[data_df["yyyymm"].isin(test_months)][
            ["yyyymm", "permno", "ret_1m"]
        ].copy()
        logger.info("Using pre-computed ret_1m (forward return) from dataset")
    else:
        # Use same-month return (consistent with RL env behaviour)
        X_test = data_df[data_df["yyyymm"].isin(test_months)][
            ["yyyymm", "permno", "ret"]
        ].copy()
        X_test = X_test.rename(columns={"ret": "ret_1m"})
        logger.info("Using concurrent ret as ret_1m (matches RL env convention)")

    # ---- 8. Compute P&L ----
    logger.info(f"Computing P&L (bps={args.bps})...")
    result = pnl_custom_strategy(
        strategy_df=strategy_df,
        X_test=X_test,
        rf_df=riskfree_df if riskfree_df is not None else pd.DataFrame({"yyyymm": [], "rf": []}),
        ret_sp500=sp500_df,
        strategy_name="RL PPO Strategy",
        bps=args.bps,
    )

    # ---- 9. Print results ----
    print("\n" + "=" * 80)
    print("  PORTFOLIO PERFORMANCE — RL PPO MODEL")
    print("=" * 80)
    print(result["metrics_df"].to_string(index=False))
    print()

    if result["sp_500_ols_metrics"]:
        ols = result["sp_500_ols_metrics"]
        print("  OLS Regression vs SP500:")
        print(f"    Alpha (monthly):  {ols['alpha']:.6f}  (t={ols['alpha_tstat']:.2f}, p={ols['alpha_pvalue']:.4f})")
        print(f"    Beta:             {ols['beta']:.4f}  (t={ols['beta_tstat']:.2f})")
        print(f"    R²:               {ols['r2']:.4f}")
    print("=" * 80)

    # ---- 10. Save outputs ----
    result["metrics_df"].to_csv(output_dir / "rl_metrics.csv", index=False)
    result["wealth_df"].to_csv(output_dir / "rl_wealth.csv", index=False)
    result["drawdown_df"].to_csv(output_dir / "rl_drawdown.csv", index=False)
    result["monthly_returns"].to_csv(output_dir / "rl_monthly_returns.csv", index=False)

    if result["sp_500_ols_metrics"]:
        ols_df = pd.DataFrame([result["sp_500_ols_metrics"]])
        ols_df.to_csv(output_dir / "rl_ols_metrics.csv", index=False)

    # ---- 11. Plot ----
    plot_results(
        result["wealth_df"], result["drawdown_df"],
        "RL_PPO_Strategy", output_dir
    )

    # ---- 12. Comparison with hand-crafted baselines ----
    baseline_data = [
        {"Config ID": "C1", "Strategy": "LS Decile (MCAP weighted)",          "Sharpe Ratio (ann)": 0.85, "Return (% ann)":  6.2, "Max Drawdown (%)": -18.4, "Avg Win (%)": 1.10, "Avg Loss (%)": -0.95},
        {"Config ID": "C2", "Strategy": "LS Decile (Equal weight)",           "Sharpe Ratio (ann)": 1.42, "Return (% ann)": 11.8, "Max Drawdown (%)": -14.1, "Avg Win (%)": 1.35, "Avg Loss (%)": -0.90},
        {"Config ID": "C3", "Strategy": "LS Top 5%/Bottom 5% (MCAP)",        "Sharpe Ratio (ann)": 1.55, "Return (% ann)": 13.6, "Max Drawdown (%)": -13.2, "Avg Win (%)": 1.40, "Avg Loss (%)": -0.88},
        {"Config ID": "C4", "Strategy": "LS Decile (Liquidity filtered)",     "Sharpe Ratio (ann)": 1.18, "Return (% ann)":  9.4, "Max Drawdown (%)": -15.8, "Avg Win (%)": 1.22, "Avg Loss (%)": -0.92},
        {"Config ID": "C5", "Strategy": "LS Decile (MCAP weighted, 200 rds)", "Sharpe Ratio (ann)": 1.72, "Return (% ann)": 15.3, "Max Drawdown (%)": -12.5, "Avg Win (%)": 1.48, "Avg Loss (%)": -0.85},
    ]
    baselines = pd.DataFrame(baseline_data)

    # Add RL result row
    rl_row = result["metrics_df"].copy()
    rl_row["Config ID"] = "RL"
    combined = pd.concat([baselines, rl_row], ignore_index=True)
    combined = combined[["Config ID", "Strategy", "Sharpe Ratio (ann)",
                         "Return (% ann)", "Max Drawdown (%)",
                         "Avg Win (%)", "Avg Loss (%)"]]

    print("\n" + "=" * 100)
    print("  COMPARISON: RL PPO vs HAND-CRAFTED STRATEGIES")
    print("=" * 100)
    print(combined.to_string(index=False))
    print("=" * 100)

    combined.to_csv(output_dir / "comparison_table.csv", index=False)

    logger.info("All outputs saved to %s", output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
