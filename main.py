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

def main(args):
    
    # Load configuration with lightweight parsing.
    config = load_config(args.config)

    # Initialize Weights & Biases for experiment tracking.
    wandb.init(
        project="ml-alpha-ranking",
        name=config.get("pipeline", {}).get("experiment_name", None),
        # name="1000 boosting rounds, no CS rank, no winsorization, median imputation, no scaling, no PCA, no centroid, Ridge on all features",
        config=config  # ton config
    )

    # Data loading and splitting
    data_manager = DataManager(config.get("data", {}))
    ret_sp500 = data_manager.get_ret_sp500(start=config["data"].get("test_start", "1990-01-01"))
    
    data_manager.get_data(start=config["data"].get("train_start", "1990-01-01"), end=config["data"].get("test_end", "2024-12-31"), market_cap=config["data"].get("market_cap", 10))
    s = data_manager.get_train_val_test(targets=["ret_1m", "ret_3m", "ret_6m"], top_n_market_cap=config["data"].get("top_market_cap", None))

    X_train, y_train, group_train = s["train"]
    X_val, y_val, group_val = s["val"]
    X_test, y_test, group_test = s["test"]

    X_train = X_train.drop(columns=["Size"])  # remove targets from train features
    X_val = X_val.drop(columns=["Size"])      # remove targets from val features
    X_test = X_test.drop(columns=["Size"])  # remove targets from test features


    # Preprocessing pipeline
    fp_cfg = config.get("feature_pipeline", {})
    feat_pipeline = FeaturePipeline(fp_cfg)
    X_train, X_val, X_test = feat_pipeline.total_preprocessing_steps(X_train=X_train, y_train=y_train, group_train=group_train, X_val=X_val, X_test=X_test)

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

    group_avg, encoded_group_mean_returns_fig = analyzer.plot_mean_realized_return_by_encoded_group(_LABEL_ENCODERS.get("decile"))

    df_long_short_test = analyzer.t_test_long_short(percentage= .1, alternative="greater")
    df_long_short_test_nw = analyzer.t_test_long_short_nw(percentage= .1, lag=3)
    df_long_short_test = pd.concat([df_long_short_test, df_long_short_test_nw], ignore_index=True)
    features_importance_figs = analyzer.get_features_importance_figures()
    history, figs = analyzer.get_history_figures()

    for target, fig in features_importance_figs.items():
        wandb.log({
            f"feature_importance/{target}": wandb.Plotly(fig)
        })

    for h, fig in zip(history.keys(), figs):
        wandb.log({
            f"history/{h}": wandb.Plotly(fig)
        })

    wandb_table = wandb.Table(dataframe=df_metrics)
    wandb.log({"test_metrics": wandb_table})
    wandb_table_long_short = wandb.Table(dataframe=df_long_short_test)
    wandb.log({"df_long_short_test": wandb_table_long_short})
    wandb_table_group_returns = wandb.Table(dataframe=group_avg)
    wandb.log({"group_returns": wandb_table_group_returns})
    wandb.log({"encoded_group_mean_returns_fig": wandb.Plotly(encoded_group_mean_returns_fig) })

    # Strategies
    rf_df = data_manager.get_rf(start=config["data"].get("test_start", "1990-01-01"))
    ret_sp500 = data_manager.get_ret_sp500(start=config["data"].get("test_start", "1990-01-01"))
    y_pred = ensemble.predict(X_test, group_test)

    X_test.to_csv("generated/X_test.csv", index=False)
    pd.DataFrame(y_pred).to_csv("generated/y_pred.csv", index=False)

    portfolio_construction = PortfolioConstruction(rf_df, X_test, y_test, y_pred)
    portfolio_analyzer = PortfolioAnalyzer(rf_df, ret_sp500, X_test, y_test)
    
    strategies = config.get("strategies", ["top_1"])
    strategy_dfs = {}
    bps = config.get("pipeline", {}).get("bps", 10)
    for strategy_name in strategies:
        print(f"Evaluating strategy: {strategy_name}")
        strategy_fn = portfolio_construction.strategies.get(strategy_name)
        if strategy_fn is None:
            raise ValueError(f"Strategy '{strategy_name}' not found in PortfolioConstruction.")
        strategy_df = strategy_fn()
        pnl, dropdown, metrics, sp_500_ols_metrics = portfolio_analyzer.pnl_custom_strategy(strategy_df, strategy_name=strategy_name, bps=bps)
        wandb.log({
            f"{strategy_name}_pnl": wandb.Plotly(pnl),
            f"{strategy_name}_drawdown": wandb.Plotly(dropdown)
        })
        wandb_table_metrics = wandb.Table(dataframe=metrics)
        wandb.log({f"{strategy_name}_metrics": wandb_table_metrics})
        wandb_table_strategy = wandb.Table(dataframe=strategy_df)
        wandb.log({f"{strategy_name}_weights": wandb_table_strategy})
        wandb_table_sp500_ols_metrics = wandb.Table(dataframe=pd.DataFrame([sp_500_ols_metrics]))
        wandb.log({f"{strategy_name}_sp500_ols_metrics": wandb_table_sp500_ols_metrics})

    # Saving results to Weights & Biases


    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate AlphaXGBoost model for stock ranking.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_francois.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    
    main(args)
