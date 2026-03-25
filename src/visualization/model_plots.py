"""Model diagnostics and ranking visualizations for the alpha ranking pipeline."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.model.model import AlphaXGBoost


def plot_training_metrics(model: AlphaXGBoost, title: str = "Training Metrics") -> go.Figure:
    """Plot training/evaluation metrics collected during model training."""
    history = model.get_training_history()
    if not history:
        raise ValueError("No training history found. Train the model with an evaluation set first.")

    fig = go.Figure()
    for dataset_name, metric_map in history.items():
        for metric_name, values in metric_map.items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(values) + 1)),
                    y=values,
                    mode="lines",
                    name=f"{dataset_name}:{metric_name}",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Boosting Iteration",
        yaxis_title="Metric Value",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )
    return fig


def plot_model_feature_importance(
    model: AlphaXGBoost,
    importance_type: str = "gain",
    top_n: int = 20,
    title: str = "Model Feature Importance",
) -> go.Figure:
    """Plot top feature importance from trained model."""
    importance = model.get_feature_importance(importance_type=importance_type)
    if not importance:
        raise ValueError("No feature importance available. Ensure the model is trained.")

    importance_df = (
        pd.DataFrame(
            [{"feature": feature, "importance": score} for feature, score in importance.items()]
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        importance_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        template="plotly_white",
    )
    fig.update_layout(height=550)
    return fig


def build_monthly_ranking_table(
    prepared_df: pd.DataFrame,
    y_pred: pd.Series | list[float],
    date_column: str = "market_data_publication_date",
    return_column: str = "monthly_return",
    stock_id_column: str = "stock_id",
    ticker_column: str = "ticker_symbol",
) -> pd.DataFrame:
    """Build full monthly ranking table (predicted rank vs actual return rank)."""
    if len(prepared_df) != len(y_pred):
        raise ValueError("prepared_df and y_pred must have the same length.")

    ranking_df = prepared_df.copy()
    ranking_df[date_column] = pd.to_datetime(ranking_df[date_column])
    ranking_df["year_month"] = ranking_df[date_column].dt.to_period("M").astype(str)
    ranking_df["predicted_score"] = list(y_pred)

    ranking_df["predicted_rank"] = ranking_df.groupby("year_month")["predicted_score"].rank(
        method="first", ascending=False
    )
    ranking_df["actual_rank"] = ranking_df.groupby("year_month")[return_column].rank(
        method="first", ascending=False
    )
    ranking_df["rank_error"] = ranking_df["predicted_rank"] - ranking_df["actual_rank"]
    ranking_df["abs_rank_error"] = ranking_df["rank_error"].abs()

    ordered_columns = [
        "year_month",
        date_column,
        stock_id_column,
        ticker_column,
        return_column,
        "predicted_score",
        "predicted_rank",
        "actual_rank",
        "rank_error",
        "abs_rank_error",
    ]

    available_columns = [col for col in ordered_columns if col in ranking_df.columns]
    return ranking_df[available_columns].sort_values(["year_month", "predicted_rank"]).reset_index(drop=True)


def plot_monthly_rank_correlation(
    ranking_df: pd.DataFrame,
    title: str = "Monthly Rank Correlation (Predicted vs Actual)",
) -> go.Figure:
    """Plot monthly Spearman correlation between predicted and actual ranks."""
    monthly_corr = (
        ranking_df.groupby("year_month")
        .apply(lambda g: g["predicted_rank"].corr(g["actual_rank"], method="spearman"))
        .reset_index(name="spearman_corr")
    )

    fig = px.line(
        monthly_corr,
        x="year_month",
        y="spearman_corr",
        title=title,
        markers=True,
        template="plotly_white",
    )
    fig.update_layout(height=500, xaxis_title="Month", yaxis_title="Spearman Correlation")
    return fig


def plot_top_k_realized_return(
    ranking_df: pd.DataFrame,
    top_k: int = 10,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot realized average return of monthly top-k predicted stocks."""
    if title is None:
        title = f"Monthly Realized Return of Top-{top_k} Predicted Stocks"

    top_k_returns = (
        ranking_df.sort_values(["year_month", "predicted_rank"])
        .groupby("year_month")
        .head(top_k)
        .groupby("year_month")["monthly_return"]
        .mean()
        .reset_index(name="avg_top_k_return")
    )

    fig = px.line(
        top_k_returns,
        x="year_month",
        y="avg_top_k_return",
        title=title,
        markers=True,
        template="plotly_white",
    )
    fig.update_layout(height=500, xaxis_title="Month", yaxis_title="Average Realized Return")
    return fig


def build_monthly_top_k_table(ranking_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Return top-k predicted stocks for each month with realized return for quick inspection."""
    return (
        ranking_df.sort_values(["year_month", "predicted_rank"])
        .groupby("year_month")
        .head(top_k)
        .reset_index(drop=True)
    )


def plot_dataframe_table(
    df: pd.DataFrame,
    title: str,
    max_rows: int = 1000,
) -> go.Figure:
    """Render a DataFrame as a Plotly HTML table for quick visual inspection."""
    display_df = df.head(max_rows).copy()

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{col}</b>" for col in display_df.columns],
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[display_df[col] for col in display_df.columns],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"{title} (showing first {len(display_df)} rows)",
        height=max(450, min(1600, 120 + 24 * len(display_df))),
    )
    return fig
