"""
Benchmark top-4 CatBoost ranking losses on FF5 idiosyncratic residuals.

Target: per-stock expanding-window OLS residual vs the 5 Fama-French factors.
  residual_it = ret_it  -  (alpha_i + beta_i · factors_t)
where beta_i is estimated on all months strictly before t (no look-ahead).
Factor realisations (Mkt_RF, SMB, HML, RMW, CMA) for month t are
contemporaneous, not future — safe to use.

Losses (same as prior benchmark):
  QueryRMSE, YetiRank, YetiRankPairwise, QuerySoftMax

Portfolio strategies per loss:
  top10_eq  — top-10%  equal weight
  top30_eq  — top-30%  equal weight
  top20_sw  — top-20%  score weighted

Precision@decile uses RESIDUALS as ground truth (idiosyncratic alpha ranking).
Portfolio PnL uses actual returns (ret_1m).

Run:
    mamba run -n mlf python benchmark_losses.py [--config config/config_francois.yaml]
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
    ("QueryRMSE",        None),
    ("YetiRank",         None),
    ("YetiRankPairwise", None),
    ("QuerySoftMax",     "top_decile"),
]

STRATEGIES = [
    ("top10_eq", "Top-10% equal weight"),
    ("top30_eq", "Top-30% equal weight"),
    ("top20_sw", "Top-20% score weighted"),
    ("top10_sv", "Top-10% score/vol weighted"),
]

VOL_MIN_MONTHS = 12   # minimum history to compute per-stock vol

FF5_MIN_MONTHS = 24   # minimum per-stock history before computing residual


# ---------------------------------------------------------------------------
# FF5 residual computation
# ---------------------------------------------------------------------------

def compute_ff5_residuals(
    permnos: np.ndarray,
    yyyymms: np.ndarray,
    rets: np.ndarray,
    ff5: pd.DataFrame,
    min_months: int = FF5_MIN_MONTHS,
) -> np.ndarray:
    """
    Per-stock expanding-window OLS residuals vs FF5 factors.

    For stock i in month t: fit OLS on months < t, then
        residual_it = ret_it - (alpha_i + beta_i · factors_t)

    Returns an array of the same length as `rets`.
    NaN where the stock has < min_months of history.
    """
    factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
    ff5_idx = ff5.set_index("yyyymm")[factor_cols]

    residuals = np.full(len(rets), np.nan)

    for permno in np.unique(permnos):
        stock_mask = permnos == permno
        raw_idx    = np.where(stock_mask)[0]
        months     = yyyymms[raw_idx]
        order      = np.argsort(months)
        idx_s      = raw_idx[order]
        months_s   = months[order]
        rets_s     = rets[idx_s]

        # Factor matrix for this stock's months (T × 5)
        fac = np.array([
            ff5_idx.loc[m].values if m in ff5_idx.index else [np.nan] * 5
            for m in months_s
        ], dtype=np.float64)

        for i in range(len(months_s)):
            if i < min_months:
                continue

            y_p = rets_s[:i]
            F_p = fac[:i]
            valid = ~np.isnan(y_p) & ~np.any(np.isnan(F_p), axis=1)
            if valid.sum() < min_months:
                continue

            # OLS: [1 | F] · coeffs = y
            X_p = np.column_stack([np.ones(valid.sum()), F_p[valid]])
            coeffs, *_ = np.linalg.lstsq(X_p, y_p[valid], rcond=None)

            curr_fac = fac[i]
            if np.any(np.isnan(curr_fac)):
                continue

            pred = coeffs[0] + curr_fac @ coeffs[1:]
            residuals[idx_s[i]] = rets_s[i] - pred

    return residuals


# ---------------------------------------------------------------------------
# Per-stock expanding-window volatility (no look-ahead)
# ---------------------------------------------------------------------------

def compute_per_stock_vol(
    permnos: np.ndarray,
    yyyymms: np.ndarray,
    rets: np.ndarray,
    min_months: int = VOL_MIN_MONTHS,
) -> dict[tuple[int, int], float]:
    """
    Per-stock expanding-window return volatility.
    Vol for (permno, month t) uses only returns from months strictly before t.
    Returns a dict {(permno, yyyymm): annualised_vol}.
    NaN entries are omitted — caller falls back to equal weight for missing stocks.
    """
    vol_map: dict[tuple[int, int], float] = {}

    for permno in np.unique(permnos):
        mask    = permnos == permno
        raw_idx = np.where(mask)[0]
        order   = np.argsort(yyyymms[raw_idx])
        idx_s   = raw_idx[order]
        months_s = yyyymms[idx_s]
        rets_s   = rets[idx_s]

        for i in range(len(months_s)):
            if i < min_months:
                continue
            past = rets_s[:i]
            valid = ~np.isnan(past)
            if valid.sum() < min_months:
                continue
            vol = float(np.std(past[valid], ddof=1) * np.sqrt(12))
            if vol > 0:
                vol_map[(int(permno), int(months_s[i]))] = vol

    return vol_map


# ---------------------------------------------------------------------------
# CatBoost pool / training helpers
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


def precision_at_decile(scores: np.ndarray, residuals: np.ndarray,
                        groups: list[int]) -> list[float]:
    """Precision@10% ranking idiosyncratic returns (residuals)."""
    precisions, cursor = [], 0
    for g in groups:
        sl = slice(cursor, cursor + g)
        k  = max(1, g // 10)
        pred_top = set(np.argsort(scores[sl])[::-1][:k])
        real_top = set(np.argsort(residuals[sl])[::-1][:k])
        precisions.append(len(pred_top & real_top) / k)
        cursor += g
    return precisions


def train_and_eval(
    loss: str, encoder_key: str | None,
    X_train: pd.DataFrame, y_train_resid: np.ndarray, groups_train: list[int],
    X_test:  pd.DataFrame, y_test_resid:  np.ndarray, groups_test:  list[int],
    num_rounds: int, depth: int, lr: float,
    subsample: float, rsm: float, seed: int,
    feat_cols: list[str],
) -> tuple[np.ndarray, list[float]] | None:
    params = dict(
        loss_function=loss, iterations=num_rounds, depth=depth,
        learning_rate=lr, bootstrap_type="Bernoulli",
        subsample=subsample, rsm=rsm,
        random_seed=seed, verbose=0, thread_count=-1,
    )
    try:
        pool_tr = build_pool(X_train, y_train_resid, groups_train, encoder_key, feat_cols)
        model   = CatBoost(params)
        model.fit(pool_tr)

        X_test_arr = X_test[feat_cols].fillna(0).values.astype(np.float32)
        scores     = model.predict(X_test_arr)
        precisions = precision_at_decile(scores, y_test_resid, groups_test)
        return scores, precisions
    except Exception:
        print(f"  FAILED ({loss}):\n{traceback.format_exc(limit=2)}")
        return None


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

def build_equal_weight(scores: np.ndarray, X_test: pd.DataFrame,
                       top_pct: float) -> pd.DataFrame:
    base = X_test[["yyyymm", "permno"]].copy()
    base["score"] = scores
    rows = []
    for yyyymm, grp in base.groupby("yyyymm"):
        k = max(1, int(np.ceil(len(grp) * top_pct)))
        top = grp.nlargest(k, "score")
        w = 1.0 / k
        for _, r in top.iterrows():
            rows.append({"yyyymm": yyyymm, "permno": r["permno"], "weight": w})
        rows.append({"yyyymm": yyyymm, "permno": -1, "weight": 0.0})
    return pd.DataFrame(rows, columns=["yyyymm", "permno", "weight"])


def build_score_weighted(scores: np.ndarray, X_test: pd.DataFrame,
                         top_pct: float) -> pd.DataFrame:
    base = X_test[["yyyymm", "permno"]].copy()
    base["score"] = scores
    rows = []
    for yyyymm, grp in base.groupby("yyyymm"):
        k = max(1, int(np.ceil(len(grp) * top_pct)))
        top = grp.nlargest(k, "score").copy()
        shifted = top["score"] - top["score"].min()
        total   = shifted.sum()
        top["weight"] = shifted / total if total > 0 else 1.0 / k
        for _, r in top.iterrows():
            rows.append({"yyyymm": yyyymm, "permno": r["permno"], "weight": r["weight"]})
        rows.append({"yyyymm": yyyymm, "permno": -1, "weight": 0.0})
    return pd.DataFrame(rows, columns=["yyyymm", "permno", "weight"])


def build_score_over_vol(scores: np.ndarray, X_test: pd.DataFrame,
                         vol_map: dict[tuple[int, int], float],
                         top_pct: float = 0.10) -> pd.DataFrame:
    """
    Top-pct% by score, weighted by shifted_score / annualised_vol.
    Stocks missing a vol estimate fall back to score-weighted.
    """
    base = X_test[["yyyymm", "permno"]].copy()
    base["score"] = scores
    rows = []
    for yyyymm, grp in base.groupby("yyyymm"):
        k   = max(1, int(np.ceil(len(grp) * top_pct)))
        top = grp.nlargest(k, "score").copy()

        shifted = top["score"] - top["score"].min()

        vols = np.array([
            vol_map.get((int(r["permno"]), int(yyyymm)), np.nan)
            for _, r in top.iterrows()
        ])

        has_vol = ~np.isnan(vols)
        raw_w   = np.where(has_vol, shifted.values / np.where(has_vol, vols, 1.0), np.nan)

        # fallback for stocks without vol: use plain score weight
        if has_vol.any():
            fallback = shifted.values / shifted.sum() if shifted.sum() > 0 else 1.0 / k
            raw_w    = np.where(has_vol, raw_w, fallback)
        else:
            raw_w = shifted.values / shifted.sum() if shifted.sum() > 0 else np.full(k, 1.0 / k)

        total = raw_w.sum()
        weights = raw_w / total if total > 0 else np.full(k, 1.0 / k)

        for (_, r), w in zip(top.iterrows(), weights):
            rows.append({"yyyymm": yyyymm, "permno": r["permno"], "weight": float(w)})
        rows.append({"yyyymm": yyyymm, "permno": -1, "weight": 0.0})
    return pd.DataFrame(rows, columns=["yyyymm", "permno", "weight"])


def make_strategy_df(key: str, scores: np.ndarray, X_test: pd.DataFrame,
                     vol_map: dict | None = None) -> pd.DataFrame:
    if key == "top10_eq":
        return build_equal_weight(scores, X_test, 0.10)
    if key == "top30_eq":
        return build_equal_weight(scores, X_test, 0.30)
    if key == "top20_sw":
        return build_score_weighted(scores, X_test, 0.20)
    if key == "top10_sv":
        if vol_map is None:
            raise ValueError("vol_map required for top10_sv strategy")
        return build_score_over_vol(scores, X_test, vol_map, top_pct=0.10)
    raise ValueError(key)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _color(i: int, n: int) -> str:
    return f"hsl({int(i / n * 240)},70%,45%)"


def plot_precision_bar(results: dict) -> go.Figure:
    rows = sorted([(loss, np.mean(p), np.std(p)) for loss, p in results.items()],
                  key=lambda x: -x[1])
    labels, means, stds = zip(*rows)
    n = len(labels)
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(means),
        error_y=dict(type="data", array=list(stds), visible=True),
        marker_color=[_color(i, n) for i in range(n)],
        text=[f"{m:.1%}" for m in means], textposition="outside",
    ))
    fig.add_hline(y=0.10, line_dash="dash", line_color="grey",
                  annotation_text="random baseline (10%)")
    fig.update_layout(title="Mean Precision@10% on FF5 residuals by loss",
                      yaxis=dict(tickformat=".0%"),
                      template="plotly_white", height=450)
    return fig


def plot_precision_box(results: dict) -> go.Figure:
    rows = sorted(results.items(), key=lambda x: -np.mean(x[1]))
    n    = len(rows)
    fig  = go.Figure()
    for i, (loss, precs) in enumerate(rows):
        fig.add_trace(go.Box(y=precs, name=loss,
                             marker_color=_color(i, n), boxmean="sd"))
    fig.add_hline(y=0.10, line_dash="dash", line_color="grey")
    fig.update_layout(title="Monthly Precision@10% distribution (FF5 residuals)",
                      yaxis=dict(tickformat=".0%"),
                      template="plotly_white", height=450)
    return fig


def plot_pnl_comparison(wealth_map: dict, strat_key: str, strat_label: str,
                        loss_names: list[str]) -> go.Figure:
    n   = len(loss_names)
    fig = go.Figure()
    for i, loss in enumerate(loss_names):
        df = wealth_map.get(loss, {}).get(strat_key)
        if df is None:
            continue
        dates = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
        fig.add_trace(go.Scatter(x=dates, y=df["wealth"], mode="lines",
                                 name=loss, line=dict(color=_color(i, n), width=2)))
    fig.update_layout(title=f"PnL — {strat_label}",
                      yaxis_title="Wealth", template="plotly_white", height=450)
    return fig


def plot_drawdown_comparison(dd_map: dict, strat_key: str, strat_label: str,
                              loss_names: list[str]) -> go.Figure:
    n   = len(loss_names)
    fig = go.Figure()
    for i, loss in enumerate(loss_names):
        df = dd_map.get(loss, {}).get(strat_key)
        if df is None:
            continue
        dates = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
        fig.add_trace(go.Scatter(x=dates, y=df["drawdown"], mode="lines",
                                 name=loss, line=dict(color=_color(i, n), width=1.5)))
    fig.update_layout(title=f"Drawdown — {strat_label}",
                      yaxis=dict(tickformat=".0%"), template="plotly_white", height=400)
    return fig


def plot_metric_grouped_bar(metrics_map: dict, metric_key: str, title: str,
                             loss_names: list[str], strat_labels: dict,
                             fmt: str = ".2f") -> go.Figure:
    fig = go.Figure()
    for strat_key, strat_label in strat_labels.items():
        values = [metrics_map.get(loss, {}).get(strat_key, {}).get(metric_key, np.nan)
                  for loss in loss_names]
        values = [v if v is not None and not np.isnan(v) else 0 for v in values]
        fig.add_trace(go.Bar(name=strat_label, x=loss_names, y=values,
                             text=[f"{v:{fmt}}" for v in values],
                             textposition="outside"))
    fig.update_layout(barmode="group", title=title,
                      template="plotly_white", height=450)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config)

    wandb.init(project="ml-alpha-ranking", name="loss-benchmark-ff5-residuals",
               config=config)

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
    # Compute FF5 residuals — expanding window, no look-ahead
    # Use train + val + test combined so test betas see all prior months
    # ------------------------------------------------------------------
    print("Computing FF5 expanding-window residuals …")
    ff5 = _load_ff5_factors()

    def _resids(X: pd.DataFrame, y_ret: pd.Series) -> np.ndarray:
        return compute_ff5_residuals(
            permnos=X["permno"].values,
            yyyymms=X["yyyymm"].values,
            rets=y_ret.values,
            ff5=ff5,
            min_months=FF5_MIN_MONTHS,
        )

    # Compute for all splits; for test/val betas, earlier months already
    # covered because we pass the same expanding window starting from train.
    # We concatenate chronologically, run once, then split back.
    all_X   = pd.concat([X_train, X_val, X_test], ignore_index=False)
    all_ret = pd.concat([y_train["ret_1m"], y_val["ret_1m"], y_test["ret_1m"]],
                        ignore_index=False)

    all_resid = compute_ff5_residuals(
        permnos=all_X["permno"].values,
        yyyymms=all_X["yyyymm"].values,
        rets=all_ret.values,
        ff5=ff5,
        min_months=FF5_MIN_MONTHS,
    )

    n_tr = len(X_train)
    n_va = len(X_val)
    resid_train_all = all_resid[:n_tr]
    resid_test_all  = all_resid[n_tr + n_va:]

    # Drop rows where residual is NaN (insufficient stock history)
    tr_valid = ~np.isnan(resid_train_all)
    te_valid = ~np.isnan(resid_test_all)

    X_train_r      = X_train.iloc[tr_valid].reset_index(drop=True)
    resid_train    = resid_train_all[tr_valid]
    groups_train_r = _recompute_groups(X_train_r["yyyymm"].values)

    X_test_r    = X_test.iloc[te_valid].reset_index(drop=True)
    resid_test  = resid_test_all[te_valid]
    groups_test_r = _recompute_groups(X_test_r["yyyymm"].values)

    nan_tr_pct = (~tr_valid).mean() * 100
    nan_te_pct = (~te_valid).mean() * 100
    print(f"  Residual NaN: train {nan_tr_pct:.1f}%  test {nan_te_pct:.1f}%")
    print(f"  Train rows after filter: {len(X_train_r)}  |  Test rows: {len(X_test_r)}")

    # ------------------------------------------------------------------
    # Per-stock expanding-window volatility (used by top10_sv strategy)
    # Run on the full chronological stack so test vols see all prior months
    # ------------------------------------------------------------------
    print("Computing per-stock expanding-window volatility …")
    vol_map = compute_per_stock_vol(
        permnos=all_X["permno"].values,
        yyyymms=all_X["yyyymm"].values,
        rets=all_ret.values,
        min_months=VOL_MIN_MONTHS,
    )
    print(f"  Vol estimates: {len(vol_map):,} (stock × month)")

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    bench_cfg  = config.get("benchmark", {})
    num_rounds = int(bench_cfg.get("num_rounds", 50))
    depth      = int(bench_cfg.get("depth", 5))
    lr         = float(bench_cfg.get("learning_rate", 0.1))
    subsample  = float(bench_cfg.get("subsample", 0.8))
    rsm        = float(bench_cfg.get("rsm", 0.8))
    seed       = int(bench_cfg.get("random_state", 42))

    loss_names   = [loss for loss, _ in LOSSES]
    strat_labels = {k: lbl for k, lbl in STRATEGIES}

    precision_results: dict[str, list[float]]            = {}
    wealth_map:        dict[str, dict[str, pd.DataFrame]] = {}
    dd_map:            dict[str, dict[str, pd.DataFrame]] = {}
    metrics_map:       dict[str, dict[str, dict]]         = {}

    for loss, encoder_key in LOSSES:
        print(f"\n=== [{loss}] encoder={encoder_key!r} ===")
        out = train_and_eval(
            loss, encoder_key,
            X_train_r, resid_train, groups_train_r,
            X_test_r,  resid_test,  groups_test_r,
            num_rounds, depth, lr, subsample, rsm, seed, feat_cols,
        )
        if out is None:
            print("  skipped.")
            continue

        scores, precisions = out
        precision_results[loss] = precisions
        print(f"  precision@10% (residuals) = {np.mean(precisions):.3f}")

        wealth_map[loss]  = {}
        dd_map[loss]      = {}
        metrics_map[loss] = {}

        for strat_key, strat_label in STRATEGIES:
            run_name = f"{loss}/{strat_key}"
            try:
                # Strategy uses filtered X_test (residual-valid rows only)
                strategy_df = make_strategy_df(strat_key, scores, X_test_r, vol_map)

                # PortfolioAnalyzer needs full X_test with ret_1m for PnL
                analyzer = PortfolioAnalyzer(rf_df, ret_sp500, X_test_r, y_test.iloc[te_valid])
                pnl_fig, dd_fig, metrics_df, sp500_ols = analyzer.pnl_custom_strategy(
                    strategy_df, strategy_name=run_name, bps=10,
                )
                ff5_metrics = analyzer.ff5_regression(run_name)

                strat_data = analyzer.all_strategy_data[run_name]
                wealth_map[loss][strat_key] = strat_data["pnl"]
                dd_map[loss][strat_key]     = strat_data["drawdown"]

                m = metrics_df.iloc[0].to_dict()
                m.update(sp500_ols)
                m.update(ff5_metrics)
                metrics_map[loss][strat_key] = m

                sharpe  = m.get("annualized_sharpe_ratio", float("nan"))
                ann_ret = m.get("annualized_return",       float("nan"))
                print(f"  {strat_label:30s}  Sharpe={sharpe:.2f}  AnnRet={ann_ret:.1%}")

                wandb.log({
                    f"bench/{loss}/{strat_key}/pnl":      wandb.Plotly(pnl_fig),
                    f"bench/{loss}/{strat_key}/drawdown": wandb.Plotly(dd_fig),
                })
            except Exception:
                print(f"  Portfolio eval FAILED ({run_name}):\n{traceback.format_exc(limit=3)}")

    if not precision_results:
        print("No losses succeeded.")
        wandb.finish()
        return

    # ------------------------------------------------------------------
    # Precision plots
    # ------------------------------------------------------------------
    wandb.log({
        "bench/precision/bar": wandb.Plotly(plot_precision_bar(precision_results)),
        "bench/precision/box": wandb.Plotly(plot_precision_box(precision_results)),
    })

    # ------------------------------------------------------------------
    # Portfolio comparison plots
    # ------------------------------------------------------------------
    for strat_key, strat_label in STRATEGIES:
        wandb.log({
            f"bench/compare/{strat_key}/pnl":      wandb.Plotly(
                plot_pnl_comparison(wealth_map, strat_key, strat_label, loss_names)),
            f"bench/compare/{strat_key}/drawdown": wandb.Plotly(
                plot_drawdown_comparison(dd_map, strat_key, strat_label, loss_names)),
        })

    for metric_key, title, fmt in [
        ("annualized_sharpe_ratio", "Annualised Sharpe by loss × strategy",  ".2f"),
        ("annualized_return",       "Annualised return by loss × strategy",   ".1%"),
        ("max_drawdown",            "Max drawdown by loss × strategy",        ".1%"),
        ("ff5_alpha",               "FF5 monthly alpha by loss × strategy",   ".4f"),
        ("ff5_alpha_tstat",         "FF5 alpha t-stat by loss × strategy",    ".2f"),
    ]:
        wandb.log({
            f"bench/compare/metrics/{metric_key}": wandb.Plotly(
                plot_metric_grouped_bar(metrics_map, metric_key, title,
                                        loss_names, strat_labels, fmt))
        })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    rows = []
    for loss, _ in LOSSES:
        for strat_key, strat_label in STRATEGIES:
            m = metrics_map.get(loss, {}).get(strat_key, {})
            rows.append({
                "loss":              loss,
                "strategy":          strat_label,
                "mean_precision":    np.mean(precision_results[loss]) if loss in precision_results else None,
                "sharpe":            m.get("annualized_sharpe_ratio"),
                "annualized_return": m.get("annualized_return"),
                "max_drawdown":      m.get("max_drawdown"),
                "ff5_alpha":         m.get("ff5_alpha"),
                "ff5_alpha_tstat":   m.get("ff5_alpha_tstat"),
                "status":            "ok" if m else "failed",
            })

    summary = pd.DataFrame(rows).sort_values(["loss", "strategy"])
    wandb.log({"bench/summary": wandb.Table(dataframe=summary)})
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_francois.yaml")
    main(parser.parse_args())
