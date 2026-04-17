import argparse

import numpy as np
import pandas as pd

from src.config.config_loader import load_config
from src.data.DataManager import DataManager
from src.data.feature_pipeline import FeaturePipeline
from src.model.model import MultiHorizonRanker, HorizonEnsemble, _LABEL_ENCODERS
from src.model.RankingAnalyzer import RankingAnalyzer
from src.portfolio.PortfolioAnalyzer import PortfolioAnalyzer
from src.portfolio.PortfolioConstruction import PortfolioConstruction

import wandb
import os
os.makedirs("generated", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_groups(X: pd.DataFrame) -> np.ndarray:
    """Compute LambdaMART group sizes from the yyyymm column.

    Data *must* already be sorted by yyyymm so that group sizes align
    with row order.
    """
    return X.groupby("yyyymm", sort=False).size().values


def _safe_to_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Write CSV with a clear error message if the file is locked."""
    try:
        df.to_csv(path, **kwargs)
    except PermissionError:
        print(
            f"\n  ERROR: Cannot write to '{path}'.\n"
            f"  Close the file in Excel / any other program, then re-run.\n"
            f"  Alternatively, delete the file manually first."
        )
        raise


# ---------------------------------------------------------------------------
# Phase 1 – Walk-Forward Expanding Window  (RL training period)
# ---------------------------------------------------------------------------

def walk_forward_predict(config, data_manager, X_all, y_all):
    """Generate OOS predictions using an expanding training window.

    Default scheme (predict_start_year=1995, predict_end_year=2015, val_years=1):
        Train 1990-1993, Val 1994 -> Predict 1995
        Train 1990-1994, Val 1995 -> Predict 1996
        ...
        Train 1990-2013, Val 2014 -> Predict 2015

    Returns
    -------
    X_oos : pd.DataFrame   - concatenated preprocessed features for OOS years
    y_oos : np.ndarray      - concatenated ensemble predictions
    """
    years = X_all["yyyymm"] // 100

    # ---- Config -----------------------------------------------------------
    wf_cfg = config.get("walk_forward", {})
    predict_start_year = wf_cfg.get("predict_start_year", 1995)
    predict_end_year   = wf_cfg.get("predict_end_year", 2015)
    val_years          = wf_cfg.get("val_years", 1)

    model_cfg   = config.get("model", {})
    verbose     = config.get("pipeline", {}).get("verbose", True)
    weights     = config.get("ensemble", {}).get("weights", None)
    combination = config.get("ensemble", {}).get("combination", "mean_rank")

    all_years_sorted = sorted(years.unique())
    pred_years = [y for y in all_years_sorted
                  if predict_start_year <= y <= predict_end_year]

    oos_X_list: list[pd.DataFrame] = []
    oos_pred_list: list[np.ndarray] = []

    for pred_year in pred_years:
        print(f"\n{'=' * 60}")
        print(f"Walk-forward: predicting {pred_year}")
        print(f"{'=' * 60}")

        val_start_year = pred_year - val_years

        train_mask = years < val_start_year
        val_mask   = (years >= val_start_year) & (years < pred_year)
        pred_mask  = years == pred_year

        n_train, n_val, n_pred = train_mask.sum(), val_mask.sum(), pred_mask.sum()
        if n_train == 0:
            print(f"  Skipping {pred_year}: no training data")
            continue
        if n_val == 0:
            print(f"  Skipping {pred_year}: no validation data")
            continue
        if n_pred == 0:
            print(f"  Skipping {pred_year}: no prediction data")
            continue

        X_tr = X_all.loc[train_mask].reset_index(drop=True)
        y_tr = y_all.loc[train_mask].reset_index(drop=True)
        g_tr = _compute_groups(X_tr)

        X_v = X_all.loc[val_mask].reset_index(drop=True)
        y_v = y_all.loc[val_mask].reset_index(drop=True)
        g_v = _compute_groups(X_v)

        X_p = X_all.loc[pred_mask].reset_index(drop=True)
        g_p = _compute_groups(X_p)

        # Fresh preprocessing fitted on this window's training data
        fp_cfg = config.get("feature_pipeline", {})
        feat_pipeline = FeaturePipeline(fp_cfg)
        X_tr_proc, X_v_proc, X_p_proc = feat_pipeline.total_preprocessing_steps(
            X_train=X_tr, y_train=y_tr, group_train=g_tr,
            X_val=X_v, X_test=X_p,
        )

        # Fresh model for this window
        model = MultiHorizonRanker(**model_cfg)
        model.fit(
            X_tr_proc, y_tr, g_tr,
            (X_v_proc, y_v), g_v,
            verbose=verbose,
        )

        ensemble = HorizonEnsemble(model, combination=combination, weights=weights)
        y_pred = ensemble.predict(X_p_proc, g_p)

        print(
            f"  Train: {n_train:,} rows  |  Val: {n_val:,} rows  "
            f"|  Pred: {n_pred:,} rows"
        )

        oos_X_list.append(X_p_proc)
        oos_pred_list.append(y_pred)

    # ---- Concatenate all OOS windows --------------------------------------
    X_oos = pd.concat(oos_X_list, ignore_index=True)
    y_oos = np.concatenate(oos_pred_list)

    print(f"\n{'=' * 60}")
    print(f"Walk-forward complete: {len(pred_years)} windows")
    print(f"  OOS rows  : {len(X_oos):,}")
    print(f"  Date range: {X_oos['yyyymm'].min()} - {X_oos['yyyymm'].max()}")
    print(f"{'=' * 60}\n")

    return X_oos, y_oos


# ---------------------------------------------------------------------------
# Phase 2 – Standard single-split  (RL val + test period)
# ---------------------------------------------------------------------------

def standard_split_predict(config, data_manager):
    """Train XGB on the full train split, predict on val AND test.

    Uses the config-defined splits (val_start / test_start etc.) so the
    XGB val/test predictions align exactly with the RL agent's eval windows.

    Returns
    -------
    X_valtest : pd.DataFrame  - preprocessed features for val + test
    y_valtest : np.ndarray     - ensemble predictions for val + test
    """
    data_manager.get_data(
        start=config["data"].get("train_start", "1990-01-01"),
        end=config["data"].get("test_end", "2024-12-31"),
        market_cap=config["data"].get("market_cap", 10),
    )

    targets = config.get("model", {}).get("targets", ["ret_1m", "ret_3m", "ret_6m"])
    s = data_manager.get_train_val_test(
        targets=targets,
        top_n_market_cap=config["data"].get("top_market_cap", None),
    )

    X_train, y_train, group_train = s["train"]
    X_val,   y_val,   group_val   = s["val"]
    X_test,  y_test,  group_test  = s["test"]

    # Preprocessing
    fp_cfg = config.get("feature_pipeline", {})
    feat_pipeline = FeaturePipeline(fp_cfg)
    X_train, X_val, X_test = feat_pipeline.total_preprocessing_steps(
        X_train=X_train, y_train=y_train, group_train=group_train,
        X_val=X_val, X_test=X_test,
    )

    # Model training (full train split, val for early stopping)
    model_cfg = config.get("model", {})
    verbose   = config.get("pipeline", {}).get("verbose", True)

    model = MultiHorizonRanker(**model_cfg)
    model.fit(X_train, y_train, group_train, (X_val, y_val), group_val, verbose=verbose)

    # Ensemble
    weights     = config.get("ensemble", {}).get("weights", None)
    combination = config.get("ensemble", {}).get("combination", "mean_rank")
    ensemble    = HorizonEnsemble(model, combination=combination, weights=weights)

    # Predict on BOTH val and test
    y_pred_val  = ensemble.predict(X_val,  group_val)
    y_pred_test = ensemble.predict(X_test, group_test)

    X_valtest = pd.concat([X_val, X_test], ignore_index=True)
    y_valtest = np.concatenate([y_pred_val, y_pred_test])

    print(f"\nStandard split predictions:")
    print(f"  Val  rows : {len(X_val):,}  "
          f"({X_val['yyyymm'].min()} - {X_val['yyyymm'].max()})")
    print(f"  Test rows : {len(X_test):,}  "
          f"({X_test['yyyymm'].min()} - {X_test['yyyymm'].max()})")
    print(f"  Total     : {len(X_valtest):,}\n")

    return X_valtest, y_valtest


# ---------------------------------------------------------------------------
# Original single-split pipeline (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def run_single_split(config, data_manager):
    """Original train / val / test pipeline with full evaluation + wandb."""

    ret_sp500 = data_manager.get_ret_sp500(
        start=config["data"].get("train_start", "1990-01-01")
    )

    data_manager.get_data(
        start=config["data"].get("train_start", "1990-01-01"),
        end=config["data"].get("test_end", "2024-12-31"),
        market_cap=config["data"].get("market_cap", 10),
    )
    s = data_manager.get_train_val_test(
        targets=["ret_1m", "ret_3m", "ret_6m"],
        top_n_market_cap=config["data"].get("top_market_cap", None),
    )

    X_train, y_train, group_train = s["train"]
    X_val, y_val, group_val = s["val"]
    X_test, y_test, group_test = s["test"]

    # Preprocessing pipeline
    fp_cfg = config.get("feature_pipeline", {})
    feat_pipeline = FeaturePipeline(fp_cfg)
    X_train, X_val, X_test = feat_pipeline.total_preprocessing_steps(
        X_train=X_train, y_train=y_train, group_train=group_train,
        X_val=X_val, X_test=X_test,
    )

    # Model training
    model_cfg = config.get("model", {})
    targets = model_cfg.get("targets", ["ret_1m", "ret_3m", "ret_6m"])
    verbose = config.get("pipeline", {}).get("verbose", True)

    model = MultiHorizonRanker(**model_cfg)
    model.fit(X_train, y_train, group_train, (X_val, y_val), group_val, verbose=verbose)

    # HorizonEnsemble
    weights = config.get("ensemble", {}).get("weights", None)
    combination = config.get("ensemble", {}).get("combination", "mean_rank")
    ensemble = HorizonEnsemble(model, combination=combination, weights=weights)

    # Ranking analysis
    label_encoder_name = config.get("model", {}).get("label_encoder", "argsort")
    eval_at = config.get("model", {}).get("eval_at", [10, 20])
    encoder_fn = _LABEL_ENCODERS.get(label_encoder_name)

    analyzer = RankingAnalyzer(model, ensemble, X_test, group_test, y_test)
    df_metrics = analyzer.evaluate(eval_at=eval_at, encoder_fn=encoder_fn)

    group_avg, encoded_group_mean_returns_fig = analyzer.plot_mean_realized_return_by_encoded_group(
        _LABEL_ENCODERS.get("decile")
    )

    df_long_short_test = analyzer.t_test_long_short(percentage=0.2, alternative="greater")
    df_long_short_test_nw = analyzer.t_test_long_short_nw(percentage=0.2, lag=3)
    df_long_short_test = pd.concat([df_long_short_test, df_long_short_test_nw], ignore_index=True)
    features_importance_figs = analyzer.get_features_importance_figures()
    history, figs = analyzer.get_history_figures()

    for target, fig in features_importance_figs.items():
        wandb.log({f"feature_importance/{target}": wandb.Plotly(fig)})

    for h, fig in zip(history.keys(), figs):
        wandb.log({f"history/{h}": wandb.Plotly(fig)})

    wandb_table = wandb.Table(dataframe=df_metrics)
    wandb.log({"test_metrics": wandb_table})
    wandb_table_long_short = wandb.Table(dataframe=df_long_short_test)
    wandb.log({"df_long_short_test": wandb_table_long_short})
    wandb_table_group_returns = wandb.Table(dataframe=group_avg)
    wandb.log({"group_returns": wandb_table_group_returns})
    wandb.log({"encoded_group_mean_returns_fig": wandb.Plotly(encoded_group_mean_returns_fig)})

    # Strategies
    rf_df = data_manager.get_rf(start=config["data"].get("test_start", "1990-01-01"))
    ret_sp500 = data_manager.get_ret_sp500(start=config["data"].get("test_start", "1990-01-01"))
    y_pred = ensemble.predict(X_test, group_test)

    X_test.to_csv("generated/X_test.csv", index=False)
    pd.DataFrame(y_pred).to_csv("generated/y_pred.csv", index=False)

    portfolio_construction = PortfolioConstruction(rf_df, X_test, y_test, y_pred)
    portfolio_analyzer = PortfolioAnalyzer(rf_df, ret_sp500, X_test, y_test)

    strategies = config.get("strategies", ["top_1"])
    bps = config.get("pipeline", {}).get("bps", 10)
    for strategy_name in strategies:
        print(f"Evaluating strategy: {strategy_name}")
        strategy_fn = portfolio_construction.strategies.get(strategy_name)
        if strategy_fn is None:
            raise ValueError(f"Strategy '{strategy_name}' not found in PortfolioConstruction.")
        strategy_df = strategy_fn()
        pnl, dropdown, metrics, sp_500_ols_metrics = portfolio_analyzer.pnl_custom_strategy(
            strategy_df, strategy_name=strategy_name, bps=bps
        )
        wandb.log({
            f"{strategy_name}_pnl": wandb.Plotly(pnl),
            f"{strategy_name}_drawdown": wandb.Plotly(dropdown),
        })
        wandb_table_metrics = wandb.Table(dataframe=metrics)
        wandb.log({f"{strategy_name}_metrics": wandb_table_metrics})
        wandb_table_strategy = wandb.Table(dataframe=strategy_df)
        wandb.log({f"{strategy_name}_weights": wandb_table_strategy})
        wandb_table_sp500_ols_metrics = wandb.Table(dataframe=pd.DataFrame([sp_500_ols_metrics]))
        wandb.log({f"{strategy_name}_sp500_ols_metrics": wandb_table_sp500_ols_metrics})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config)

    wandb.init(
        project="ml-alpha-ranking",
        config=config,
    )

    data_manager = DataManager(config.get("data", {}))

    if args.walk_forward:
        # ==================================================================
        # HYBRID MODE
        #   Phase 1: Walk-forward OOS for RL training   (1995 - 2015)
        #   Phase 2: Standard XGB preds for RL val/test (2016 - end)
        # ==================================================================

        # --- Load full dataset once for the walk-forward pool --------------
        data_manager.get_data(
            start=config["data"].get("train_start", "1990-01-01"),
            end=config["data"].get("test_end", "2024-12-31"),
            market_cap=config["data"].get("market_cap", 10),
        )
        targets = config.get("model", {}).get("targets", ["ret_1m", "ret_3m", "ret_6m"])
        s = data_manager.get_train_val_test(
            targets=targets,
            top_n_market_cap=config["data"].get("top_market_cap", None),
        )

        X_all = pd.concat(
            [s["train"][0], s["val"][0], s["test"][0]], ignore_index=True
        )
        y_all = pd.concat(
            [s["train"][1], s["val"][1], s["test"][1]], ignore_index=True
        )
        # Sort by yyyymm for consistent group computation
        sort_idx = X_all["yyyymm"].argsort()
        X_all = X_all.iloc[sort_idx].reset_index(drop=True)
        y_all = y_all.iloc[sort_idx].reset_index(drop=True)

        # --- Phase 1: walk-forward 1995 -> 2015 ---------------------------
        print("\n" + "=" * 60)
        print("PHASE 1  -  Walk-forward expanding window (RL training)")
        print("=" * 60)
        X_wf, y_wf = walk_forward_predict(config, data_manager, X_all, y_all)

        # --- Phase 2: standard split for val + test -----------------------
        print("\n" + "=" * 60)
        print("PHASE 2  -  Standard XGB split (RL val + test)")
        print("=" * 60)
        X_vt, y_vt = standard_split_predict(config, data_manager)

        # --- Concatenate phases -------------------------------------------
        X_combined = pd.concat([X_wf, X_vt], ignore_index=True)
        y_combined = np.concatenate([y_wf, y_vt])

        print(f"\nCombined output:")
        print(f"  Total rows : {len(X_combined):,}")
        print(f"  Date range : {X_combined['yyyymm'].min()} - "
              f"{X_combined['yyyymm'].max()}")
        print(f"  Phase 1 (WF)  : {len(X_wf):,} rows  "
              f"({X_wf['yyyymm'].min()} - {X_wf['yyyymm'].max()})")
        print(f"  Phase 2 (Std) : {len(X_vt):,} rows  "
              f"({X_vt['yyyymm'].min()} - {X_vt['yyyymm'].max()})")

        # --- Save ----------------------------------------------------------
        _safe_to_csv(X_combined, "generated/X_test.csv", index=False)
        _safe_to_csv(pd.DataFrame(y_combined), "generated/y_pred.csv", index=False)

        print("\nSaved generated/X_test.csv and generated/y_pred.csv")
        print("Run export_alpha_signal.py to produce the RL alpha signal.")

    else:
        # ==================================================================
        # Original single-split pipeline with full evaluation
        # ==================================================================
        run_single_split(config, data_manager)

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train and evaluate AlphaXGBoost model for stock ranking."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_francois.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help=(
            "Enable hybrid walk-forward mode. "
            "Phase 1: expanding-window OOS for RL training (1995-2015). "
            "Phase 2: standard XGB val+test predictions for RL eval (2016+). "
            "Configure via the 'walk_forward' key in the YAML config."
        ),
    )
    args = parser.parse_args()

    main(args)