"""Data visualization and exploratory data analysis for portfolio data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_stocks_per_month(
    df: pd.DataFrame,
    date_column: str = "market_data_publication_date",
    title: str = "Number of Stocks Per Month",
) -> go.Figure:
    """
    Plot the number of active stocks per month over time.

    Args:
        df: Input DataFrame
        date_column: Column name for date
        title: Plot title

    Returns:
        Plotly figure
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy["year_month"] = df_copy[date_column].dt.to_period("M")

    stocks_per_month = df_copy.groupby("year_month")["stock_id"].nunique().reset_index()
    stocks_per_month["year_month"] = stocks_per_month["year_month"].astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stocks_per_month["year_month"],
            y=stocks_per_month["stock_id"],
            mode="lines+markers",
            name="Number of Stocks",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Number of Stocks",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        showlegend=False,
    )

    return fig


def plot_monthly_return_distribution(
    df: pd.DataFrame,
    target_col: str = "monthly_return",
    title: str = "Distribution of Monthly Returns",
) -> go.Figure:
    """
    Plot histogram and statistics of monthly returns.

    Args:
        df: Input DataFrame
        target_col: Target column name
        title: Plot title

    Returns:
        Plotly figure
    """
    df_clean = df[df[target_col].notna()].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df_clean[target_col],
            nbinsx=50,
            name="Monthly Returns",
            marker_color="rgba(31, 119, 180, 0.7)",
            hovertemplate="Return Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    # Add mean and median lines
    mean_return = df_clean[target_col].mean()
    median_return = df_clean[target_col].median()

    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_return:.4f}",
        annotation_position="top right",
    )

    fig.add_vline(
        x=median_return,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_return:.4f}",
        annotation_position="top",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Monthly Return",
        yaxis_title="Frequency",
        template="plotly_white",
        height=500,
        showlegend=False,
    )

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str = "monthly_return",
    exclude_cols: Optional[list[str]] = None,
    title: str = "Correlation with Monthly Returns",
) -> go.Figure:
    """
    Plot correlation heatmap of features with target.

    Args:
        df: Input DataFrame
        target_col: Target column name
        exclude_cols: Columns to exclude from features
        title: Plot title

    Returns:
        Plotly figure
    """
    if exclude_cols is None:
        exclude_cols = [
            "stock_id",
            "ticker_symbol",
            "market_data_publication_date",
            "publication_date",
            "fiscal_quarter_end_date",
        ]

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Remove NaN and inf values
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).dropna()

    # Select features
    feature_cols = [col for col in df_numeric.columns if col not in exclude_cols and col != target_col]

    # Calculate correlations with target
    correlations = df_numeric[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    correlations = correlations.sort_values(ascending=True)

    fig = go.Figure(
        data=go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation="h",
            marker=dict(
                color=correlations.values,
                colorscale="RdBu",
                cmid=0,
                colorbar=dict(title="Correlation"),
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Correlation Coefficient",
        yaxis_title="Features",
        template="plotly_white",
        height=600,
        showlegend=False,
    )

    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    n_features: int = 6,
    exclude_cols: Optional[list[str]] = None,
    title: str = "Distribution of Key Features",
) -> go.Figure:
    """
    Plot distributions of top numeric features (ranked by variance).

    Args:
        df: Input DataFrame
        n_features: Number of features to plot
        exclude_cols: Columns to exclude
        title: Plot title

    Returns:
        Plotly figure
    """
    if exclude_cols is None:
        exclude_cols = [
            "stock_id",
            "ticker_symbol",
            "market_data_publication_date",
            "publication_date",
            "fiscal_quarter_end_date",
            "monthly_return",
        ]

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Remove NaN and inf values
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).dropna()

    # Select top features by variance
    feature_cols = [col for col in df_numeric.columns if col not in exclude_cols]
    variances = df_numeric[feature_cols].var().sort_values(ascending=False)
    top_features = variances.head(n_features).index.tolist()

    # Create subplots
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=top_features,
        specs=[[{"secondary_y": False}] * n_cols] * n_rows,
    )

    for idx, feature in enumerate(top_features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        fig.add_trace(
            go.Histogram(
                x=df_numeric[feature],
                nbinsx=30,
                name=feature,
                marker_color="rgba(31, 119, 180, 0.7)",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="Value", row=n_rows, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=200 * n_rows,
        showlegend=False,
    )

    return fig


def plot_return_over_time(
    df: pd.DataFrame,
    date_column: str = "market_data_publication_date",
    target_col: str = "monthly_return",
    title: str = "Average Monthly Return Over Time",
) -> go.Figure:
    """
    Plot average monthly return over time.

    Args:
        df: Input DataFrame
        date_column: Column name for date
        target_col: Target column name
        title: Plot title

    Returns:
        Plotly figure
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy["year_month"] = df_copy[date_column].dt.to_period("M")

    # Calculate average return per month
    avg_return = df_copy.groupby("year_month")[target_col].agg(["mean", "std", "count"]).reset_index()
    avg_return["year_month"] = avg_return["year_month"].astype(str)

    fig = go.Figure()

    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=avg_return["year_month"],
            y=avg_return["mean"],
            mode="lines+markers",
            name="Average Return",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        )
    )

    # Add confidence bands (mean +/- std)
    fig.add_trace(
        go.Scatter(
            x=avg_return["year_month"],
            y=avg_return["mean"] + avg_return["std"],
            fill=None,
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=avg_return["year_month"],
            y=avg_return["mean"] - avg_return["std"],
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="Std Dev Range",
            fillcolor="rgba(31, 119, 180, 0.2)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Average Monthly Return",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_data_coverage(
    df: pd.DataFrame,
    date_column: str = "market_data_publication_date",
    title: str = "Data Coverage Over Time",
) -> go.Figure:
    """
    Plot data coverage (percentage of non-null values) over time.

    Args:
        df: Input DataFrame
        date_column: Column name for date
        title: Plot title

    Returns:
        Plotly figure
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy["year_month"] = df_copy[date_column].dt.to_period("M")

    # Calculate coverage per month for key numeric columns
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    key_features = [col for col in numeric_cols if col not in ["stock_id", "monthly_return"]][:5]

    fig = go.Figure()

    for feature in key_features:
        coverage = (
            df_copy.groupby("year_month")[feature]
            .apply(lambda x: (x.notna().sum() / len(x)) * 100)
            .reset_index()
        )
        coverage["year_month"] = coverage["year_month"].astype(str)

        fig.add_trace(
            go.Scatter(
                x=coverage["year_month"],
                y=coverage[feature],
                mode="lines",
                name=feature,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Data Coverage (%)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def generate_summary_statistics(
    df: pd.DataFrame,
    target_col: str = "monthly_return",
    exclude_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate summary statistics for all numeric columns.

    Args:
        df: Input DataFrame
        target_col: Target column name (for visibility)
        exclude_cols: Columns to exclude

    Returns:
        DataFrame with summary statistics
    """
    if exclude_cols is None:
        exclude_cols = [
            "stock_id",
            "ticker_symbol",
            "market_data_publication_date",
            "publication_date",
            "fiscal_quarter_end_date",
        ]

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Remove NaN and inf values for statistics
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

    feature_cols = [col for col in df_numeric.columns if col not in exclude_cols]

    stats = df_numeric[feature_cols].describe().T
    stats["non_null_count"] = df_numeric[feature_cols].count()
    stats["null_count"] = df_numeric[feature_cols].isna().sum()
    stats["null_pct"] = (stats["null_count"] / len(df_numeric)) * 100
    stats["skewness"] = df_numeric[feature_cols].skew()
    stats["kurtosis"] = df_numeric[feature_cols].kurtosis()

    return stats[["count", "non_null_count", "null_count", "null_pct", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness", "kurtosis"]]


def plot_summary_statistics(
    stats_df: pd.DataFrame,
    title: str = "Summary Statistics Table",
) -> go.Figure:
    """
    Create a plotly table from summary statistics DataFrame.

    Args:
        stats_df: DataFrame with summary statistics
        title: Plot title

    Returns:
        Plotly Figure with table
    """
    # Add explicit feature names (index) as a column for readability.
    display_df = stats_df.copy()
    display_df.insert(0, "feature", display_df.index)
    display_df = display_df.reset_index(drop=True).round(4)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>" + col + "</b>" for col in display_df.columns],
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
        title=title,
        height=max(400, len(display_df) * 25 + 100),
        showlegend=False,
    )

    return fig


def plot_price_vs_volume(
    df: pd.DataFrame,
    title: str = "Price vs Volume Scatter",
) -> go.Figure:
    """
    Scatter plot of price vs volume (sample to avoid overplotting).

    Args:
        df: Input DataFrame
        title: Plot title

    Returns:
        Plotly figure
    """
    df_sample = df[["price", "volume", "monthly_return"]].dropna().sample(min(5000, len(df)))

    fig = px.scatter(
        df_sample,
        x="price",
        y="volume",
        color="monthly_return",
        title=title,
        labels={"price": "Stock Price", "volume": "Trading Volume", "monthly_return": "Monthly Return"},
        color_continuous_scale="RdBu",
        opacity=0.6,
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
    )

    return fig


def plot_return_percentiles(
    df: pd.DataFrame,
    target_col: str = "monthly_return",
    title: str = "Monthly Return Percentiles Over Time",
) -> go.Figure:
    """
    Plot percentiles of returns over time (p10, p25, p50, p75, p90).

    Args:
        df: Input DataFrame
        target_col: Target column name
        title: Plot title

    Returns:
        Plotly figure
    """
    df_copy = df.copy()
    df_copy["market_data_publication_date"] = pd.to_datetime(df_copy["market_data_publication_date"])
    df_copy["year_month"] = df_copy["market_data_publication_date"].dt.to_period("M")

    percentiles = df_copy.groupby("year_month")[target_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    percentiles.index = percentiles.index.astype(str)

    fig = go.Figure()

    # Add median
    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles[0.5],
            mode="lines",
            name="Median (p50)",
            line=dict(width=2),
        )
    )

    # Add p75-p25 band
    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles[0.75],
            fill=None,
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles[0.25],
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="IQR (p25-p75)",
            fillcolor="rgba(0, 100, 200, 0.2)",
        )
    )

    # Add p90-p10 band
    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles[0.9],
            fill=None,
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles[0.1],
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="90% Range (p10-p90)",
            fillcolor="rgba(0, 100, 200, 0.1)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Monthly Return",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_stock_ranking_evolution(
    df: pd.DataFrame,
    ticker_symbol: str,
    date_column: str = "market_data_publication_date",
    target_col: str = "monthly_return",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot the ranking evolution of a single stock over time.
    
    For each month, the stock is ranked among all stocks in that month
    (based on monthly_return), with rank 1 being the best performer.

    Args:
        df: Input DataFrame
        ticker_symbol: Ticker symbol to track (e.g., 'TSLA')
        date_column: Column name for date
        target_col: Target column name (for ranking)
        title: Plot title (defaults to stock name)

    Returns:
        Plotly figure
    """
    if title is None:
        title = f"Ranking Evolution: {ticker_symbol}"
    
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy["year_month"] = df_copy[date_column].dt.to_period("M")
    
    # Filter for the given ticker
    stock_data = df_copy[df_copy["ticker_symbol"] == ticker_symbol].copy()
    
    if stock_data.empty:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text=f"No data found for ticker {ticker_symbol}")
        return fig
    
    # Rank stocks within each month (ascending ranking: 1 is best)
    # Handle NaN returns by excluding them from ranking
    monthly_rankings = []
    
    for month, group in df_copy.groupby("year_month"):
        # Only rank rows with non-null returns
        group_valid = group[group[target_col].notna()].copy()
        
        if group_valid.empty:
            continue
        
        # Rank in descending order of return (higher return = lower rank number)
        group_valid["rank"] = group_valid[target_col].rank(method="min", ascending=False)
        
        # Extract the rank for this stock in this month
        stock_row = group_valid[group_valid["ticker_symbol"] == ticker_symbol]
        
        if not stock_row.empty:
            rank = stock_row["rank"].values[0]
            monthly_rankings.append({
                "year_month": month.astype(str),
                "rank": rank,
                "return": stock_row[target_col].values[0],
            })
    
    if not monthly_rankings:
        fig = go.Figure()
        fig.add_annotation(text=f"No ranking data found for ticker {ticker_symbol}")
        return fig
    
    ranking_df = pd.DataFrame(monthly_rankings)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=ranking_df["year_month"],
            y=ranking_df["rank"],
            mode="lines+markers",
            name=ticker_symbol,
            line=dict(width=2),
            marker=dict(size=6),
            text=ranking_df["return"],
            hovertemplate="<b>%{x}</b><br>Rank: %{y}<br>Return: %{text:.4f}<extra></extra>",
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Rank (1 = best performer)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        yaxis=dict(autorange="reversed"),  # Invert y-axis so rank 1 is at top
    )
    
    return fig
