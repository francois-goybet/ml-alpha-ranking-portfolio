"""Model diagnostics and ranking visualizations for the alpha ranking pipeline."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_history(history):
    """
    Returns a list of figures (one per horizon).
    Each figure:
    - 1 row
    - 2 cols (train / eval)
    - all metrics plotted together
    """

    splits = ["train", "eval"]

    horizons = list(history.keys())

    # detect metrics safely
    sample_h = horizons[0]
    metrics = list(history[sample_h]["train"].keys())

    figures = []

    for h in horizons:

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f"{h} - train", f"{h} - eval"]
        )

        for j, s in enumerate(splits):

            for m in metrics:
                y = history[h][s][m]
                x = list(range(len(y)))

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=m,
                        showlegend=(j == 0)  # legend once per figure
                    ),
                    row=1,
                    col=j + 1
                )

        fig.update_layout(
            height=450,
            width=1000,
            title=f"History - {h}",
            autosize=False,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h")
        )

        fig.update_xaxes(matches=None)
        fig.update_yaxes(range=[0, 1],matches=None)
        fig.update_layout(
            autosize=False,
            height=450,
            width=1000,
            margin=dict(l=60, r=30, t=60, b=50)
        )
        figures.append(fig)

    return figures

def plot_feature_importance(importance, target):
    df_features_importance = pd.DataFrame({
        "feature": importance[0],
        "importance": importance[1]
    })

    df_features_importance = df_features_importance.sort_values(
        by="importance",
        ascending=False
    ).head(10)

    fig = px.bar(
        df_features_importance,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Feature Importance for {target}"
    )

    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        height=600,  
        margin=dict(l=150, r=50, t=50, b=50)  
    )

    return fig


def plot_encoded_group_mean_returns(df_group_returns: pd.DataFrame, title: str = "Mean Realized Return by Encoded Group"):
    fig = px.bar(
        df_group_returns,
        x="encoded_group",
        y="mean_realized_return",
        title=title,
        labels={
            "encoded_group": "Encoded group",
            "mean_realized_return": "Average monthly mean realized return",
        },
    )

    fig.update_layout(
        height=500,
        xaxis_type="category",
    )

    return fig