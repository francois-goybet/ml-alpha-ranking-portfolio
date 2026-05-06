"""
Characteristic-sorted (decile) portfolio analysis.

Standard academic evaluation methodology (cf. Gu, Kelly & Xiu 2020):
  1. Each month, sort stocks into N deciles by model predicted score
  2. Form equal-weighted portfolios within each decile
  3. Evaluate: cumulative PnL, mean return monotonicity, Sharpe, FF5 alpha

Trains on FF5 idiosyncratic residuals (expanding-window betas, no look-ahead).
Portfolio PnL uses actual returns (ret_1m).
No transaction costs — this is a signal evaluation framework.

Losses: QueryRMSE and YetiRank (best two from prior benchmarks).

Run:
    mamba run -n mlf python characteristic_portfolios.py [--config config/config_francois.yaml]
"""

import argparse
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import wandb
from catboost import CatBoost, Pool as CatPool

from src.config.config_loader import load_config
from src.data.DataManager import DataManager
from src.data.feature_pipeline import FeaturePipeline
from src.model.model import _LABEL_ENCODERS, _META, _recompute_groups
from src.portfolio.PortfolioAnalyzer import _load_ff5_factors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOSSES = [
    ("QueryRMSE", None),
    ("YetiRank",  None),
]

N_DECILES    = 10
FF5_MIN_MONTHS = 24


# ---------------------------------------------------------------------------
# FF5 residuals
# ---------------------------------------------------------------------------

def compute_ff5_residuals(
    permnos: np.ndarray, yyyymms: np.ndarray, rets: np.ndarray,
    ff5: pd.DataFrame, min_months: int = FF5_MIN_MONTHS,
) -> np.ndarray:
    factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
    ff5_idx     = ff5.set_index("yyyymm")[factor_cols]
    residuals   = np.full(len(rets), np.nan)

    for permno in np.unique(permnos):
        mask   = permnos == permno
        raw_idx = np.where(mask)[0]
        order  = np.argsort(yyyymms[raw_idx])
        idx_s  = raw_idx[order]
        months_s = yyyymms[idx_s]
        rets_s = rets[idx_s]

        fac = np.array([
            ff5_idx.loc[m].values if m in ff5_idx.index else [np.nan] * 5
            for m in months_s
        ], dtype=np.float64)

        for i in range(len(months_s)):
            if i < min_months:
                continue
            y_p   = rets_s[:i]
            F_p   = fac[:i]
            valid = ~np.isnan(y_p) & ~np.any(np.isnan(F_p), axis=1)
            if valid.sum() < min_months:
                continue
            X_p = np.column_stack([np.ones(valid.sum()), F_p[valid]])
            coeffs, *_ = np.linalg.lstsq(X_p, y_p[valid], rcond=None)
            curr_fac = fac[i]
            if np.any(np.isnan(curr_fac)):
                continue
            residuals[idx_s[i]] = rets_s[i] - (coeffs[0] + curr_fac @ coeffs[1:])

    return residuals


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    loss: str, encoder_key: str | None,
    X_train: pd.DataFrame, y_resid: np.ndarray, groups: list[int],
    feat_cols: list[str],
    num_rounds: int, depth: int, lr: float,
    subsample: float, rsm: float, seed: int,
) -> CatBoost | None:
    params = dict(
        loss_function=loss, iterations=num_rounds, depth=depth,
        learning_rate=lr, bootstrap_type="Bernoulli",
        subsample=subsample, rsm=rsm,
        random_seed=seed, verbose=0, thread_count=-1,
    )
    try:
        X_arr    = X_train[feat_cols].fillna(0).values.astype(np.float32)
        group_id = np.repeat(np.arange(len(groups)), groups).astype(np.uint32)
        if encoder_key is not None:
            enc   = _LABEL_ENCODERS[encoder_key]
            y_arr = enc(pd.Series(y_resid), groups).astype(np.float32)
        else:
            y_arr = y_resid.astype(np.float32)
        pool  = CatPool(data=X_arr, label=y_arr, group_id=group_id,
                        feature_names=feat_cols)
        model = CatBoost(params)
        model.fit(pool)
        return model
    except Exception:
        print(f"  Training FAILED ({loss}):\n{traceback.format_exc(limit=2)}")
        return None


# ---------------------------------------------------------------------------
# Decile portfolio construction
# ---------------------------------------------------------------------------

def build_decile_returns(
    scores: np.ndarray,
    X_test: pd.DataFrame,
    ret_col: str = "ret_1m",
    n_deciles: int = N_DECILES,
) -> pd.DataFrame:
    """
    For each month, assign stocks to deciles by score (1=lowest, N=highest),
    compute equal-weighted return per decile.

    Returns a DataFrame: index=yyyymm, columns=decile labels 1..N.
    """
    base = X_test[["yyyymm", ret_col]].copy()
    base["score"] = scores

    rows = []
    for yyyymm, grp in base.groupby("yyyymm"):
        grp = grp.dropna(subset=[ret_col, "score"])
        if len(grp) < n_deciles:
            continue
        # qcut assigns decile label 1..N (lowest score = 1)
        labels = pd.qcut(grp["score"], q=n_deciles,
                         labels=range(1, n_deciles + 1), duplicates="drop")
        grp = grp.copy()
        grp["decile"] = labels.values
        monthly = grp.groupby("decile")[ret_col].mean()
        row = {"yyyymm": yyyymm}
        for d in range(1, n_deciles + 1):
            row[d] = monthly.get(d, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("yyyymm").sort_index()
    df.columns = [int(c) for c in df.columns]
    return df


def build_ls_returns(decile_ret: pd.DataFrame,
                     long_decile: int = N_DECILES,
                     short_decile: int = 1) -> pd.Series:
    """Long top decile, short bottom decile (zero-cost)."""
    return (decile_ret[long_decile] - decile_ret[short_decile]).rename("L/S")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def cumulative_wealth(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.std(ddof=1) == 0 or len(r) < 2:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    wealth = float((1 + r).prod())
    return wealth ** (12 / len(r)) - 1


def max_drawdown(returns: pd.Series) -> float:
    w = cumulative_wealth(returns)
    return float((w / w.cummax() - 1).min())


def ff5_alpha(returns: pd.Series, ff5: pd.DataFrame) -> dict:
    """Regress monthly excess returns on FF5 factors. Returns alpha & t-stat."""
    factors = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
    r = returns.dropna().rename("ret")
    df = r.reset_index()
    df["yyyymm"] = df["yyyymm"].astype(int)
    df = df.merge(ff5[["yyyymm"] + factors + ["RF"]], on="yyyymm", how="inner")
    df["excess"] = df["ret"] - df["RF"]
    if len(df) < 10:
        return {"ff5_alpha": np.nan, "ff5_alpha_tstat": np.nan, "ff5_r2": np.nan}
    X = sm.add_constant(df[factors].astype(float))
    result = sm.OLS(df["excess"].astype(float), X).fit()
    return {
        "ff5_alpha":       result.params["const"],
        "ff5_alpha_tstat": result.tvalues["const"],
        "ff5_r2":          result.rsquared,
    }


def decile_summary(decile_ret: pd.DataFrame, ff5: pd.DataFrame) -> pd.DataFrame:
    """Per-decile: mean return, Sharpe, annualised return, max drawdown, FF5 alpha."""
    rows = []
    ls = build_ls_returns(decile_ret)
    for d in list(decile_ret.columns) + ["L/S"]:
        series = ls if d == "L/S" else decile_ret[d]
        m = {
            "decile":            d,
            "mean_monthly_ret":  float(series.mean()),
            "sharpe":            sharpe(series),
            "annualized_return": annualized_return(series),
            "max_drawdown":      max_drawdown(series),
        }
        m.update(ff5_alpha(series, ff5))
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

DECILE_COLORSCALE = [
    f"hsl({int(i / (N_DECILES - 1) * 120)},70%,45%)"
    for i in range(N_DECILES)
]   # red (D1) → green (D10)


def plot_decile_pnl(decile_ret: pd.DataFrame, loss: str) -> go.Figure:
    """Cumulative wealth for each decile + L/S overlay."""
    fig = go.Figure()
    months = pd.to_datetime(decile_ret.index.astype(str), format="%Y%m")

    for i, d in enumerate(decile_ret.columns):
        wealth = cumulative_wealth(decile_ret[d])
        fig.add_trace(go.Scatter(
            x=months, y=wealth, mode="lines",
            name=f"D{d}",
            line=dict(color=DECILE_COLORSCALE[i], width=1.5),
        ))

    ls = build_ls_returns(decile_ret)
    ls_wealth = cumulative_wealth(ls)
    fig.add_trace(go.Scatter(
        x=months, y=ls_wealth, mode="lines",
        name="L/S (D10−D1)",
        line=dict(color="black", width=2.5, dash="dash"),
    ))

    fig.update_layout(
        title=f"Decile cumulative wealth — {loss}",
        yaxis_title="Wealth (start=1)",
        template="plotly_white", height=500,
    )
    return fig


def plot_monotonicity(summary: pd.DataFrame, loss: str) -> go.Figure:
    """Bar chart of mean monthly return per decile — should be monotone."""
    rows = summary[summary["decile"] != "L/S"].copy()
    rows["decile"] = rows["decile"].astype(int)
    rows = rows.sort_values("decile")

    fig = go.Figure(go.Bar(
        x=[f"D{d}" for d in rows["decile"]],
        y=rows["mean_monthly_ret"],
        marker_color=DECILE_COLORSCALE,
        text=[f"{v:.2%}" for v in rows["mean_monthly_ret"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="grey", line_dash="dash")
    fig.update_layout(
        title=f"Mean monthly return by decile — {loss}",
        yaxis=dict(tickformat=".1%"),
        template="plotly_white", height=420,
    )
    return fig


def plot_metric_bar(summary: pd.DataFrame, metric: str,
                    title: str, fmt: str, loss: str) -> go.Figure:
    """Bar chart of any metric per decile + L/S."""
    rows = summary.copy()
    # put L/S last
    rows["_order"] = rows["decile"].apply(
        lambda x: 11 if x == "L/S" else int(x)
    )
    rows = rows.sort_values("_order")

    colors = DECILE_COLORSCALE + ["black"]
    labels = [f"D{d}" if d != "L/S" else "L/S" for d in rows["decile"]]

    fig = go.Figure(go.Bar(
        x=labels, y=rows[metric],
        marker_color=colors,
        text=[f"{v:{fmt}}" if pd.notna(v) else "" for v in rows[metric]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="grey", line_dash="dash")
    fig.update_layout(
        title=f"{title} — {loss}",
        template="plotly_white", height=420,
    )
    return fig


def plot_return_heatmap(decile_ret: pd.DataFrame, loss: str) -> go.Figure:
    """Month × decile heatmap of realised returns."""
    matrix = decile_ret.values.T   # shape (N_deciles, n_months)
    months = [str(m) for m in decile_ret.index]
    decile_labels = [f"D{d}" for d in decile_ret.columns]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=months, y=decile_labels,
        colorscale="RdYlGn", zmid=0,
        colorbar=dict(title="Monthly ret", tickformat=".0%"),
    ))
    fig.update_layout(
        title=f"Return heatmap (month × decile) — {loss}",
        template="plotly_white",
        height=max(350, 40 * N_DECILES),
    )
    return fig


def plot_ls_comparison(ls_series: dict[str, pd.Series]) -> go.Figure:
    """Overlay L/S cumulative wealth for all losses."""
    palette = {"QueryRMSE": "hsl(220,70%,45%)", "YetiRank": "hsl(30,70%,45%)"}
    fig = go.Figure()
    for loss, series in ls_series.items():
        months = pd.to_datetime(series.index.astype(str), format="%Y%m")
        fig.add_trace(go.Scatter(
            x=months, y=cumulative_wealth(series), mode="lines",
            name=loss, line=dict(color=palette.get(loss, "grey"), width=2.5),
        ))
    fig.update_layout(
        title="Long-short (D10−D1) comparison across losses",
        yaxis_title="Wealth", template="plotly_white", height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config)

    wandb.init(project="ml-alpha-ranking",
               name="characteristic-sorted-portfolios", config=config)

    data_cfg = config.get("data", {})
    dm = DataManager(data_cfg)
    dm.get_data(
        start=data_cfg.get("train_start", "1990-01-01"),
        end=data_cfg.get("test_end", "2024-12-31"),
        market_cap=data_cfg.get("market_cap", 10),
    )
    s = dm.get_train_val_test(
        targets=["ret_1m"],
        top_n_market_cap=data_cfg.get("top_market_cap", None),
    )
    X_train, y_train, g_train = s["train"]
    X_val,   y_val,   g_val   = s["val"]
    X_test,  y_test,  g_test  = s["test"]

    fp = FeaturePipeline(config.get("feature_pipeline", {}))
    X_train, X_val, X_test = fp.total_preprocessing_steps(
        X_train=X_train, y_train=y_train, group_train=g_train,
        X_val=X_val, X_test=X_test,
    )
    feat_cols = [c for c in X_train.columns if c not in _META]

    # ------------------------------------------------------------------
    # FF5 residuals
    # ------------------------------------------------------------------
    print("Computing FF5 expanding-window residuals …")
    ff5 = _load_ff5_factors()

    all_X   = pd.concat([X_train, X_val, X_test], ignore_index=False)
    all_ret = pd.concat(
        [y_train["ret_1m"], y_val["ret_1m"], y_test["ret_1m"]], ignore_index=False,
    )
    all_resid = compute_ff5_residuals(
        all_X["permno"].values, all_X["yyyymm"].values,
        all_ret.values, ff5, min_months=FF5_MIN_MONTHS,
    )

    n_tr = len(X_train)
    n_va = len(X_val)
    resid_train_all = all_resid[:n_tr]
    resid_test_all  = all_resid[n_tr + n_va:]

    tr_valid = ~np.isnan(resid_train_all)
    te_valid = ~np.isnan(resid_test_all)

    X_train_r      = X_train.iloc[tr_valid].reset_index(drop=True)
    resid_train    = resid_train_all[tr_valid]
    groups_train_r = _recompute_groups(X_train_r["yyyymm"].values)

    X_test_r = X_test.iloc[te_valid].reset_index(drop=True)
    print(f"  NaN filtered: train {(~tr_valid).mean()*100:.1f}%  "
          f"test {(~te_valid).mean()*100:.1f}%")

    # ------------------------------------------------------------------
    # Benchmark params
    # ------------------------------------------------------------------
    bench_cfg  = config.get("benchmark", {})
    num_rounds = int(bench_cfg.get("num_rounds", 50))
    depth      = int(bench_cfg.get("depth", 5))
    lr         = float(bench_cfg.get("learning_rate", 0.1))
    subsample  = float(bench_cfg.get("subsample", 0.8))
    rsm        = float(bench_cfg.get("rsm", 0.8))
    seed       = int(bench_cfg.get("random_state", 42))

    ls_series_all: dict[str, pd.Series] = {}

    for loss, encoder_key in LOSSES:
        print(f"\n=== [{loss}] ===")
        model = train_model(
            loss, encoder_key,
            X_train_r, resid_train, groups_train_r, feat_cols,
            num_rounds, depth, lr, subsample, rsm, seed,
        )
        if model is None:
            continue

        X_arr  = X_test_r[feat_cols].fillna(0).values.astype(np.float32)
        scores = model.predict(X_arr)

        # Decile portfolios
        decile_ret = build_decile_returns(scores, X_test_r, ret_col="ret_1m")
        summary    = decile_summary(decile_ret, ff5)
        ls         = build_ls_returns(decile_ret)
        ls_series_all[loss] = ls

        summary["decile"] = summary["decile"].astype(str)
        print(summary[["decile", "mean_monthly_ret", "sharpe",
                        "annualized_return", "ff5_alpha", "ff5_alpha_tstat"]]
              .to_string(index=False))

        # Plots
        fig_pnl   = plot_decile_pnl(decile_ret, loss)
        fig_mono  = plot_monotonicity(summary, loss)
        fig_heat  = plot_return_heatmap(decile_ret, loss)
        fig_sharpe = plot_metric_bar(
            summary, "sharpe", "Sharpe ratio by decile", ".2f", loss)
        fig_alpha  = plot_metric_bar(
            summary, "ff5_alpha", "FF5 monthly alpha by decile", ".4f", loss)
        fig_tstat  = plot_metric_bar(
            summary, "ff5_alpha_tstat", "FF5 alpha t-stat by decile", ".2f", loss)
        fig_annret = plot_metric_bar(
            summary, "annualized_return", "Annualised return by decile", ".1%", loss)

        wandb.log({
            f"decile/{loss}/pnl":              wandb.Plotly(fig_pnl),
            f"decile/{loss}/monotonicity":     wandb.Plotly(fig_mono),
            f"decile/{loss}/heatmap":          wandb.Plotly(fig_heat),
            f"decile/{loss}/sharpe":           wandb.Plotly(fig_sharpe),
            f"decile/{loss}/ff5_alpha":        wandb.Plotly(fig_alpha),
            f"decile/{loss}/ff5_alpha_tstat":  wandb.Plotly(fig_tstat),
            f"decile/{loss}/annualized_return":wandb.Plotly(fig_annret),
            f"decile/{loss}/summary":          wandb.Table(dataframe=summary),
        })

    # Cross-loss L/S comparison
    if len(ls_series_all) > 1:
        wandb.log({
            "decile/ls_comparison": wandb.Plotly(plot_ls_comparison(ls_series_all))
        })

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_francois.yaml")
    main(parser.parse_args())
