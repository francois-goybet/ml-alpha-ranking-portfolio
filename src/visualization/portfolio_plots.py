import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_pnl(df: pd.DataFrame, title: str):
    fig = go.Figure()

    df = df.copy()
    df["date"] = pd.to_datetime(
        df["yyyymm"].astype(int).astype(str),
        format="%Y%m"
    )

    # W&B SAFE: convert to string
    df["date_str"] = df["date"].dt.strftime("%Y-%m")

    fig.add_trace(go.Scatter(
        x=df["date_str"],
        y=df["wealth"],
        mode="lines+markers",
        name=title
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Wealth ($)",
        height=500
    )

    fig.update_xaxes(
        type="category",   # important for W&B
        tickangle=45
    )

    return fig