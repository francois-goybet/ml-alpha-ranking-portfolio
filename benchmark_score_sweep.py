"""
Score-weighted portfolio sweep: top-pct% from 5% to 50% in 5% steps.

Trains once per loss on FF5 residuals (expanding-window betas, no look-ahead),
then sweeps the portfolio inclusion threshold for score-weighted strategies.

Losses: QueryRMSE (best Sharpe overall) and YetiRank (best drawdown control).

Run:
    mamba run -n mlf python benchmark_score_sweep.py [--config config/config_francois.yaml]
"""

import argparse
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from catboost import CatBoost, Pool as CatPool

from src.config.config_loader import load_config
from src.data.DataManager import DataManager
from src.data.feature_pipeline import FeaturePipeline
from src.model.model import _LABEL_ENCODERS, _META, _recompute_groups
from src.portfolio.PortfolioAnalyzer import PortfolioAnalyzer, _load_ff5_factors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOSSES = [
    ("QueryRMSE", None),
    ("YetiRank",  None),
]

TOP_PCTS = [p / 100 for p in range(5, 55, 5)]   # 5%, 10%, …, 50%
FF5_MIN_MONTHS = 24


# ---------------------------------------------------------------------------
# FF5 residuals (same as benchmark_losses.py)
# ---------------------------------------------------------------------------

def compute_ff5_residuals(
    permnos: np.ndarray,
    yyyymms: np.ndarray,
    rets: np.ndarray,
    ff5: pd.DataFrame,
    min_months: int = FF5_MIN_MONTHS,
) -> np.ndarray:
    factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
    ff5_idx     = ff5.set_index("yyyymm")[factor_cols]
    residuals   = np.full(len(rets), np.nan)

    for permno in np.unique(permnos):
        stock_mask = permnos == permno
        raw_idx    = np.where(stock_mask)[0]
        months     = yyyymms[raw_idx]
        order      = np.argsort(months)
        idx_s      = raw_idx[order]
        months_s   = months[order]
        rets_s     = rets[idx_s]

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

def build_pool(X: pd.DataFrame, y: np.ndarray, groups: list[int],
               encoder_key: str | None, feat_cols: list[str]) -> CatPool:
    X_arr    = X[feat_cols].fillna(0).values.astype(np.float32)
    group_id = np.repeat(np.arange(len(groups)), groups).astype(np.uint32)
    if encoder_key is not None:
        enc   = _LABEL_ENCODERS[encoder_key]
        y_arr = enc(pd.Series(y), groups).astype(np.float32)
    else:
        y_arr = y.astype(np.float32)
    return CatPool(data=X_arr, label=y_arr, group_id=group_id,
                   feature_names=feat_cols)


def train_model(loss: str, encoder_key: str | None,
                X_train: pd.DataFrame, y_resid: np.ndarray, groups: list[int],
                feat_cols: list[str],
                num_rounds: int, depth: int, lr: float,
                subsample: float, rsm: float, seed: int) -> CatBoost | None:
    params = dict(
        loss_function=loss, iterations=num_rounds, depth=depth,
        learning_rate=lr, bootstrap_type="Bernoulli",
        subsample=subsample, rsm=rsm,
        random_seed=seed, verbose=0, thread_count=-1,
    )
    try:
        pool = build_pool(X_train, y_resid, groups, encoder_key, feat_cols)
        model = CatBoost(params)
        model.fit(pool)
        return model
    except Exception:
        print(f"  Training FAILED ({loss}):\n{traceback.format_exc(limit=2)}")
        return None


# ---------------------------------------------------------------------------
# Strategy builder
# ---------------------------------------------------------------------------

def build_score_weighted(scores: np.ndarray, X_test: pd.DataFrame,
                         top_pct: float) -> pd.DataFrame:
    base = X_test[["yyyymm", "permno"]].copy()
    base["score"] = scores
    rows = []
    for yyyymm, grp in base.groupby("yyyymm"):
        k       = max(1, int(np.ceil(len(grp) * top_pct)))
        top     = grp.nlargest(k, "score").copy()
        shifted = top["score"] - top["score"].min()
        total   = shifted.sum()
        top["weight"] = shifted / total if total > 0 else 1.0 / k
        for _, r in top.iterrows():
            rows.append({"yyyymm": yyyymm, "permno": r["permno"], "weight": r["weight"]})
        rows.append({"yyyymm": yyyymm, "permno": -1, "weight": 0.0})
    return pd.DataFrame(rows, columns=["yyyymm", "permno", "weight"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _loss_color(loss: str) -> str:
    palette = {"QueryRMSE": "hsl(220,70%,45%)", "YetiRank": "hsl(30,70%,45%)"}
    return palette.get(loss, "grey")


def plot_metric_vs_pct(sweep: dict[str, dict[float, dict]],
                        metric_key: str, title: str,
                        y_fmt: str = ".2f",
                        hline: float | None = None) -> go.Figure:
    """Line plot: x = top_pct, y = metric, one line per loss."""
    fig = go.Figure()
    for loss, pct_map in sweep.items():
        pcts   = sorted(pct_map.keys())
        values = [pct_map[p].get(metric_key, np.nan) for p in pcts]
        fig.add_trace(go.Scatter(
            x=[f"{int(p*100)}%" for p in pcts],
            y=values, mode="lines+markers",
            name=loss,
            line=dict(color=_loss_color(loss), width=2),
            marker=dict(size=6),
        ))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=title,
        xaxis_title="Top-pct included (score-weighted)",
        template="plotly_white", height=420,
    )
    return fig


def plot_pnl_grid(wealth_map: dict[str, dict[float, pd.DataFrame]],
                  highlight_pcts: list[float] | None = None) -> go.Figure:
    """Overlay PnL curves; highlight selected pcts with solid lines, rest dashed."""
    if highlight_pcts is None:
        highlight_pcts = [0.10, 0.20, 0.30, 0.50]
    fig = go.Figure()
    for loss, pct_map in wealth_map.items():
        for pct, df in sorted(pct_map.items()):
            dates = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
            is_hl = pct in highlight_pcts
            label = f"{loss} {int(pct*100)}%"
            base_color = _loss_color(loss)
            fig.add_trace(go.Scatter(
                x=dates, y=df["wealth"],
                mode="lines", name=label,
                line=dict(color=base_color,
                          width=2.5 if is_hl else 1,
                          dash="solid" if is_hl else "dot"),
                opacity=1.0 if is_hl else 0.4,
            ))
    fig.update_layout(title="PnL sweep — score-weighted, all pcts",
                      yaxis_title="Wealth", template="plotly_white", height=500)
    return fig


def plot_pnl_best(wealth_map: dict[str, dict[float, pd.DataFrame]],
                  sweep: dict[str, dict[float, dict]]) -> go.Figure:
    """PnL for the best pct per loss (by Sharpe) only."""
    fig = go.Figure()
    for loss, pct_map in sweep.items():
        best_pct = max(pct_map, key=lambda p: pct_map[p].get("annualized_sharpe_ratio", -np.inf))
        df = wealth_map[loss].get(best_pct)
        if df is None:
            continue
        dates = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
        sharpe = pct_map[best_pct].get("annualized_sharpe_ratio", float("nan"))
        fig.add_trace(go.Scatter(
            x=dates, y=df["wealth"], mode="lines",
            name=f"{loss} {int(best_pct*100)}% (Sharpe={sharpe:.2f})",
            line=dict(color=_loss_color(loss), width=2.5),
        ))
    fig.update_layout(title="PnL — best pct per loss (by Sharpe)",
                      yaxis_title="Wealth", template="plotly_white", height=450)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config)

    wandb.init(project="ml-alpha-ranking", name="score-weighted-sweep", config=config)

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

    rf_df     = dm.get_rf(start=data_cfg.get("test_start", "2019-01-01"))
    ret_sp500 = dm.get_ret_sp500(start=data_cfg.get("test_start", "2019-01-01"))

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
        [y_train["ret_1m"], y_val["ret_1m"], y_test["ret_1m"]], ignore_index=False
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

    X_test_r      = X_test.iloc[te_valid].reset_index(drop=True)
    groups_test_r = _recompute_groups(X_test_r["yyyymm"].values)
    y_test_r      = y_test.iloc[te_valid].reset_index(drop=True)

    print(f"  NaN filtered: train {(~tr_valid).mean()*100:.1f}%  "
          f"test {(~te_valid).mean()*100:.1f}%")

    # ------------------------------------------------------------------
    # Train once per loss, then sweep pcts
    # ------------------------------------------------------------------
    bench_cfg  = config.get("benchmark", {})
    num_rounds = int(bench_cfg.get("num_rounds", 50))
    depth      = int(bench_cfg.get("depth", 5))
    lr         = float(bench_cfg.get("learning_rate", 0.1))
    subsample  = float(bench_cfg.get("subsample", 0.8))
    rsm        = float(bench_cfg.get("rsm", 0.8))
    seed       = int(bench_cfg.get("random_state", 42))

    # sweep[loss][top_pct] = metrics dict
    sweep:      dict[str, dict[float, dict]]         = {}
    wealth_map: dict[str, dict[float, pd.DataFrame]] = {}

    for loss, encoder_key in LOSSES:
        print(f"\n=== Training [{loss}] ===")
        model = train_model(
            loss, encoder_key,
            X_train_r, resid_train, groups_train_r,
            feat_cols, num_rounds, depth, lr, subsample, rsm, seed,
        )
        if model is None:
            continue

        X_arr  = X_test_r[feat_cols].fillna(0).values.astype(np.float32)
        scores = model.predict(X_arr)

        sweep[loss]      = {}
        wealth_map[loss] = {}

        for top_pct in TOP_PCTS:
            pct_label = f"{int(top_pct*100)}%"
            run_name  = f"{loss}/sw_{pct_label}"
            try:
                strategy_df = build_score_weighted(scores, X_test_r, top_pct)
                analyzer    = PortfolioAnalyzer(rf_df, ret_sp500, X_test_r, y_test_r)
                pnl_fig, dd_fig, metrics_df, sp500_ols = analyzer.pnl_custom_strategy(
                    strategy_df, strategy_name=run_name, bps=10,
                )
                ff5_m = analyzer.ff5_regression(run_name)

                strat_data = analyzer.all_strategy_data[run_name]
                wealth_map[loss][top_pct] = strat_data["pnl"]

                m = metrics_df.iloc[0].to_dict()
                m.update(sp500_ols)
                m.update(ff5_m)
                sweep[loss][top_pct] = m

                sharpe  = m.get("annualized_sharpe_ratio", float("nan"))
                ann_ret = m.get("annualized_return",       float("nan"))
                mdd     = m.get("max_drawdown",            float("nan"))
                print(f"  sw {pct_label:>4s}  Sharpe={sharpe:.2f}  "
                      f"AnnRet={ann_ret:.1%}  MDD={mdd:.1%}")

                wandb.log({
                    f"sweep/{loss}/sw_{pct_label}/pnl":      wandb.Plotly(pnl_fig),
                    f"sweep/{loss}/sw_{pct_label}/drawdown": wandb.Plotly(dd_fig),
                })
            except Exception:
                print(f"  FAILED ({run_name}):\n{traceback.format_exc(limit=3)}")

    if not sweep:
        print("No results.")
        wandb.finish()
        return

    # ------------------------------------------------------------------
    # Metric vs pct line plots
    # ------------------------------------------------------------------
    for metric_key, title, fmt, hline in [
        ("annualized_sharpe_ratio", "Sharpe vs portfolio breadth (score-weighted)",     ".2f",  None),
        ("annualized_return",       "Annualised return vs portfolio breadth",            ".1%",  None),
        ("max_drawdown",            "Max drawdown vs portfolio breadth",                 ".1%",  None),
        ("ff5_alpha",               "FF5 monthly alpha vs portfolio breadth",            ".4f",  0.0),
        ("ff5_alpha_tstat",         "FF5 alpha t-stat vs portfolio breadth",             ".2f",  2.0),
    ]:
        wandb.log({
            f"sweep/metrics/{metric_key}": wandb.Plotly(
                plot_metric_vs_pct(sweep, metric_key, title, fmt, hline))
        })

    # PnL overlay and best-pct summary
    wandb.log({
        "sweep/pnl_grid": wandb.Plotly(plot_pnl_grid(wealth_map)),
        "sweep/pnl_best": wandb.Plotly(plot_pnl_best(wealth_map, sweep)),
    })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    rows = []
    for loss, pct_map in sweep.items():
        for top_pct, m in sorted(pct_map.items()):
            rows.append({
                "loss":              loss,
                "top_pct":           f"{int(top_pct*100)}%",
                "sharpe":            m.get("annualized_sharpe_ratio"),
                "annualized_return": m.get("annualized_return"),
                "max_drawdown":      m.get("max_drawdown"),
                "ff5_alpha":         m.get("ff5_alpha"),
                "ff5_alpha_tstat":   m.get("ff5_alpha_tstat"),
            })

    summary = pd.DataFrame(rows)
    wandb.log({"sweep/summary": wandb.Table(dataframe=summary)})
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_francois.yaml")
    main(parser.parse_args())
