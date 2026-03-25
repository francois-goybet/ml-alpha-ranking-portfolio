import argparse
import os
import numpy as np
import pandas as pd

from src.config.config_loader import load_and_validate_config
from src.data.DataManager import DataManager
from src.model.model import AlphaXGBoost
from src.visualization import data_plots
from src.visualization import model_plots


def print_section(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}")
    print(title)
    print(line)


def print_saved(filepath: str) -> None:
    print(f"  [saved] {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate alpha ranking model.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main():
    """Train and evaluate an XGBoost learning-to-rank model for stock ranking.

    The script runs in clearly separated parts:
    1) Config and data split
    2) Feature preparation
    3) Data EDA plots
    4) Model training and diagnostics
    5) Validation preview
    6) Test ranking exports
    """

    # ---------------------------------------------------------------------
    # Part 1 - Load configuration and split data
    # ---------------------------------------------------------------------
    args = parse_args()
    config = load_and_validate_config(args.config)

    data_config = config["data_manager"]
    model_config = config["model"]
    pipeline_config = config.get("pipeline", {})

    # 1. Initialize DataManager and load data
    manager = DataManager(data_config)

    print_section("PART 1 - DATA SPLIT")
    train_df, val_df, test_df = manager.split_by_period()
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # ---------------------------------------------------------------------
    # Part 2 - Prepare ranking inputs
    # ---------------------------------------------------------------------
    print_section("PART 2 - PREPARE RANKING INPUTS")

    print("[train] building X, y, groups")
    X_train, y_train, groups_train = manager.prepare_ranking_data(train_df)
    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Number of months in training: {len(groups_train)}")
    print(f"Avg stocks per month: {np.mean(groups_train):.1f}")
    print(f"Feature columns: {X_train.columns.tolist()}")

    print("\n[val] building X, y, groups")
    X_val, y_val, groups_val = manager.prepare_ranking_data(val_df)
    print(f"Validation features shape: {X_val.shape}")
    print(f"Number of months in validation: {len(groups_val)}")

    # ---------------------------------------------------------------------
    # Part 3 - Generate data exploratory plots
    # ---------------------------------------------------------------------
    print_section("PART 3 - DATA PLOTS")
    plots_dir = pipeline_config.get("plots_dir", "generated/plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Use raw data for some plots, split data for others
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 1. Stocks per month
    fig_stocks = data_plots.plot_stocks_per_month(all_df)
    path_stocks = f"{plots_dir}/01_stocks_per_month.html"
    fig_stocks.write_html(path_stocks)
    print_saved(path_stocks)

    # 2. Monthly return distribution
    fig_dist = data_plots.plot_monthly_return_distribution(all_df)
    path_dist = f"{plots_dir}/02_return_distribution.html"
    fig_dist.write_html(path_dist)
    print_saved(path_dist)

    # 3. Return percentiles over time
    fig_percentiles = data_plots.plot_return_percentiles(all_df)
    path_percentiles = f"{plots_dir}/03_return_percentiles.html"
    fig_percentiles.write_html(path_percentiles)
    print_saved(path_percentiles)

    # 4. Correlation heatmap
    fig_corr = data_plots.plot_correlation_heatmap(train_df)
    path_corr = f"{plots_dir}/04_correlation_heatmap.html"
    fig_corr.write_html(path_corr)
    print_saved(path_corr)

    # 5. Feature distributions
    fig_features = data_plots.plot_feature_distributions(train_df, n_features=6)
    path_features = f"{plots_dir}/05_feature_distributions.html"
    fig_features.write_html(path_features)
    print_saved(path_features)

    # 6. Average return over time
    fig_return_time = data_plots.plot_return_over_time(all_df)
    path_return_time = f"{plots_dir}/06_return_over_time.html"
    fig_return_time.write_html(path_return_time)
    print_saved(path_return_time)

    # 7. Data coverage
    fig_coverage = data_plots.plot_data_coverage(train_df)
    path_coverage = f"{plots_dir}/07_data_coverage.html"
    fig_coverage.write_html(path_coverage)
    print_saved(path_coverage)

    # 8. Price vs Volume
    fig_pv = data_plots.plot_price_vs_volume(train_df)
    path_pv = f"{plots_dir}/08_price_vs_volume.html"
    fig_pv.write_html(path_pv)
    print_saved(path_pv)

    # 9. Summary statistics
    stats_df = data_plots.generate_summary_statistics(train_df)
    path_stats_csv = f"{plots_dir}/09_summary_statistics.csv"
    stats_df.to_csv(path_stats_csv)
    fig_stats = data_plots.plot_summary_statistics(stats_df)
    path_stats_table = f"{plots_dir}/09_summary_statistics_table.html"
    fig_stats.write_html(path_stats_table)
    print_saved(path_stats_csv)
    print_saved(path_stats_table)

    print(f"Data plots directory: {plots_dir}")

    # ---------------------------------------------------------------------
    # Part 4 - Train model and model diagnostics
    # ---------------------------------------------------------------------
    print_section("PART 4 - MODEL TRAINING")
    model = AlphaXGBoost(model_config)

    model.fit(
        X_train,
        y_train,
        groups=groups_train,
        eval_set=(X_val, y_val),
        eval_groups=groups_val,
        verbose=False,
    )
    print("Model training completed")

    model_plots_dir = f"{plots_dir}/model"
    os.makedirs(model_plots_dir, exist_ok=True)

    fig_training_metrics = model_plots.plot_training_metrics(model)
    path_training_metrics = f"{model_plots_dir}/01_training_metrics.html"
    fig_training_metrics.write_html(path_training_metrics)
    print_saved(path_training_metrics)

    fig_model_importance = model_plots.plot_model_feature_importance(model, importance_type="gain", top_n=20)
    path_model_importance = f"{model_plots_dir}/02_feature_importance_gain.html"
    fig_model_importance.write_html(path_model_importance)
    print_saved(path_model_importance)

    # ---------------------------------------------------------------------
    # Part 5 - Validation preview
    # ---------------------------------------------------------------------
    print_section("PART 5 - VALIDATION PREVIEW")
    y_pred_val = model.predict(X_val)
    print(f"Validation predictions shape: {y_pred_val.shape}")
    print(f"Prediction range: [{y_pred_val.min():.4f}, {y_pred_val.max():.4f}]")

    print("\nExample: first validation month ranking")
    top_k_preview = pipeline_config.get("top_k_preview", 10)
    month_size = groups_val[0]
    month_idx = slice(0, month_size)

    month_stocks = val_df.iloc[month_idx][["stock_id", "ticker_symbol", "monthly_return"]].copy()
    month_stocks["predicted_score"] = y_pred_val[month_idx]
    month_stocks["rank"] = month_stocks["predicted_score"].rank(ascending=False)
    month_stocks = month_stocks.sort_values("rank").head(top_k_preview)

    print(f"Top {top_k_preview} predicted stocks (first validation month):")
    print(month_stocks[["ticker_symbol", "monthly_return", "predicted_score", "rank"]])

    print("\nFeature importance snapshot (weight)")
    feature_importance = model.get_feature_importance(importance_type="weight")
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 features:")
    for feat, importance in top_features:
        print(f"  {feat}: {importance}")

    # ---------------------------------------------------------------------
    # Part 6 - Test ranking outputs and diagnostics
    # ---------------------------------------------------------------------
    print_section("PART 6 - TEST RANKING OUTPUTS")
    X_test, y_test, groups_test = manager.prepare_ranking_data(test_df)
    prepared_test_df = manager.get_last_prepared_frame()
    print(f"Test set features shape: {X_test.shape}")
    y_pred_test = model.predict(X_test)
    print(f"Test predictions generated for {len(y_pred_test)} stocks")
    print(f"Number of test months: {len(groups_test)}")

    test_ranking_df = model_plots.build_monthly_ranking_table(
        prepared_df=prepared_test_df,
        y_pred=y_pred_test,
        date_column=data_config.get("date_column", "market_data_publication_date"),
        return_column=data_config.get("target_col", "monthly_return"),
    )
    path_test_ranking_full = f"{model_plots_dir}/03_test_monthly_ranking_full.csv"
    test_ranking_df.to_csv(path_test_ranking_full, index=False)
    print_saved(path_test_ranking_full)

    fig_test_ranking_full_table = model_plots.plot_dataframe_table(
        test_ranking_df,
        title="Test Monthly Ranking - Full Table",
        max_rows=1000,
    )
    path_test_ranking_full_html = f"{model_plots_dir}/03_test_monthly_ranking_full.html"
    fig_test_ranking_full_table.write_html(path_test_ranking_full_html)
    print_saved(path_test_ranking_full_html)

    test_top_k_df = model_plots.build_monthly_top_k_table(test_ranking_df, top_k=top_k_preview)
    path_test_top_k = f"{model_plots_dir}/04_test_monthly_top_k.csv"
    test_top_k_df.to_csv(path_test_top_k, index=False)
    print_saved(path_test_top_k)

    fig_test_top_k_table = model_plots.plot_dataframe_table(
        test_top_k_df,
        title=f"Test Monthly Top-{top_k_preview} - Table",
        max_rows=1000,
    )
    path_test_top_k_html = f"{model_plots_dir}/04_test_monthly_top_k.html"
    fig_test_top_k_table.write_html(path_test_top_k_html)
    print_saved(path_test_top_k_html)

    fig_rank_corr = model_plots.plot_monthly_rank_correlation(test_ranking_df)
    path_rank_corr = f"{model_plots_dir}/05_test_monthly_rank_correlation.html"
    fig_rank_corr.write_html(path_rank_corr)
    print_saved(path_rank_corr)

    fig_top_k_return = model_plots.plot_top_k_realized_return(test_ranking_df, top_k=top_k_preview)
    path_top_k_return = f"{model_plots_dir}/06_test_top_k_realized_return.html"
    fig_top_k_return.write_html(path_top_k_return)
    print_saved(path_top_k_return)

    print_section("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()