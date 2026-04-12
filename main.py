import argparse

import numpy as np
import pandas as pd

from src.config.config_loader import load_config
from src.data.DataManager import DataManager
from src.data.feature_pipeline import FeaturePipeline
from src.model.model import MultiHorizonRanker, HorizonEnsemble, _LABEL_ENCODERS
from src.model.RankingAnalyzer import RankingAnalyzer

import wandb

def main(args):

    # Load configuration with lightweight parsing.
    config = load_config(args.config)

    # Initialize Weights & Biases for experiment tracking.
    wandb.init(
        project="ml-alpha-ranking",
        # name="1000 boosting rounds, no CS rank, no winsorization, median imputation, no scaling, no PCA, no centroid, Ridge on all features",
        config=config  # ton config
    )

    # Data loading and splitting
    data_manager = DataManager(config.get("data", {}))
    data_manager.get_data(start=config["data"].get("train_start", "1990-01-01"), end=config["data"].get("test_end", "2024-12-31"), market_cap=config["data"].get("market_cap", 10))
    s = data_manager.get_train_val_test(targets=["ret_1m", "ret_3m", "ret_6m"])

    X_train, y_train, group_train = s["train"]
    X_val, y_val, group_val = s["val"]
    X_test, y_test, group_test = s["test"]

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
    history, figs = analyzer.get_history_figures()

    # Saving results to Weights & Biases
    for h, fig in zip(history.keys(), figs):
        wandb.log({
            f"history/{h}": wandb.Plotly(fig)
        })

    wandb_table = wandb.Table(dataframe=df_metrics)
    wandb.log({"test_metrics": wandb_table})
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
