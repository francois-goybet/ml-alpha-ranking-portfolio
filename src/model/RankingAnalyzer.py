from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from src.model.model import MultiHorizonRanker, HorizonEnsemble
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

from src.visualization.model_plots import (
    plot_encoded_group_mean_returns,
    plot_feature_importance,
    plot_history,
)

"""RankingAnalyzer for evaluating multi-horizon ranking models."""

class RankingAnalyzer:
    """
    Analyzes predictions from a MultiHorizonRanker and HorizonEnsemble.
    
    Args:
        model: A fitted MultiHorizonRanker instance.
        ensemble: A fitted HorizonEnsemble instance.
        X_test: Test feature matrix (DataFrame).
        group_test: List ofw stock counts per month in test set.
        y_test: Test target DataFrame with columns matching model.targets.
    """
    
    def __init__(
        self,
        model: MultiHorizonRanker,
        ensemble: HorizonEnsemble,
        X_test: pd.DataFrame,
        group_test: list[int],
        y_test: pd.DataFrame,
    ):
        if not model.is_fitted:
            raise ValueError("model must be fitted before creating RankingAnalyzer.")
        if not ensemble.multi_ranker.is_fitted:
            raise ValueError("ensemble's multi_ranker must be fitted.")
        
        self.model = model
        self.ensemble = ensemble
        self.X_test = X_test
        self.group_test = group_test
        self.y_test = y_test
        
        # Validate shapes
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have the same length.")
        if sum(group_test) != len(X_test):
            raise ValueError("sum(group_test) must equal len(X_test).")
    
    def evaluate(self, eval_at: list[int], encoder_fn: Optional[callable] = None) -> pd.DataFrame:
        """
        Evaluates both the multi-horizon ranker and the ensemble.
        Returns a DataFrame with metrics for each target and the ensemble.
        """
        df_multi = self.evaluate_multi_horizon_ranker(eval_at, encoder_fn)
        df_ensemble = self.evaluate_ensemble(eval_at, encoder_fn)
        return pd.concat([df_multi, df_ensemble], ignore_index=True)

    def evaluate_multi_horizon_ranker(self, eval_at: list[int], encoder_fn: Optional[callable] = None) -> pd.DataFrame:
        """
        Evaluates model and ensemble predictions using AUC, NDCG@k, and HIT@k.
        Args:
            eval_at: List of k values for NDCG@k and HIT@k evaluation.
            encoder_fn: Optional function to encode continuous targets to integer grades.
        Returns:
            DataFrame with evaluation metrics for each target and the ensemble.
        """
        rows = []
        predictions = self.model.predict(self.X_test)

        for target, scores in predictions.items():
            y_t = self.y_test[target]
            groups = list(self.group_test)

            encoded = encoder_fn(y_t, groups) if encoder_fn else y_t.to_numpy()

            ndcg_scores = {k: [] for k in eval_at}
            hit_scores = {k: [] for k in eval_at}

            cursor = 0
            for g in groups:
                sl = slice(cursor, cursor + g)
                for k in eval_at:
                    ndcg_scores[k].append(self.ndcg_at_k(scores[sl], encoded[sl].astype(float), k))
                    hit_scores[k].append(self.precision_at_k(scores[sl], encoded[sl].astype(float), k))
                cursor += g

            auc = roc_auc_score((encoded > np.median(encoded)).astype(int), scores)

            row = {
                "target": target,
                "AUC": auc,
            }

            for k in eval_at:
                row[f"NDCG@{k}"] = np.mean(ndcg_scores[k])
                row[f"HIT@{k}"] = np.mean(hit_scores[k])

            rows.append(row)

        return pd.DataFrame(rows)
    
    def evaluate_ensemble(self, eval_at: list[int], encoder_fn: Optional[callable] = None) -> pd.DataFrame:
        """
        Evaluates ensemble predictions using AUC, NDCG@k, and HIT@k.
        Args:
            eval_at: List of k values for NDCG@k and HIT@k evaluation.
            encoder_fn: Optional function to encode continuous targets to integer grades.   
        Returns:
            DataFrame with evaluation metrics for the ensemble.
        """

        ensemble_scores = self.ensemble.predict(self.X_test, groups=list(self.group_test))


        ref_target = list(self.y_test.columns)[0]
        y_ref = self.y_test[ref_target]
        ref_encoded = encoder_fn(y_ref, list(self.group_test)) if encoder_fn else y_ref.to_numpy()

        ndcg_scores = {k: [] for k in eval_at}
        hit_scores = {k: [] for k in eval_at}

        cursor = 0
        for g in self.group_test:
            sl = slice(cursor, cursor + g)
            for k in eval_at:
                ndcg_scores[k].append(self.ndcg_at_k(ensemble_scores[sl], ref_encoded[sl].astype(float), k))
                hit_scores[k].append(self.precision_at_k(ensemble_scores[sl], ref_encoded[sl].astype(float), k))
            cursor += g

        auc_ens = roc_auc_score((ref_encoded > np.median(ref_encoded)).astype(int), ensemble_scores)

        row = {
            "target": f"ensemble/{self.ensemble.combination}",
            "AUC": auc_ens,
        }

        for k in eval_at:
            row[f"NDCG@{k}"] = np.mean(ndcg_scores[k])
            row[f"HIT@{k}"] = np.mean(hit_scores[k])

        return pd.DataFrame([row])
    
    def get_history_figures(self):
        history = self.model.get_history()
        # Ensemble models return empty histories — skip plotting
        has_curves = any(bool(v) for v in history.values())
        figs = plot_history(history) if has_curves else []
        return history, figs

    def get_features_importance_figures(self):
        importance = self.model.get_feature_importance()
        figs = {}
        for target, features_importance in importance.items():  
            x = features_importance.keys()
            y = features_importance.values()
            fig = plot_feature_importance((x,y), target)
            figs[target] = fig
        return figs

    def plot_mean_realized_return_by_encoded_group(
        self,
        encoder_fn: callable,
    ):
        """
        Build a bar plot where:
        - x-axis: encoded group (from encoder_fn applied to ensemble predictions)
        - y-axis: mean over months of each month's mean realized return within group
        """

        y_pred = self.ensemble.predict(self.X_test, groups=list(self.group_test))

        encoded_pred = encoder_fn(pd.Series(y_pred), list(self.group_test))
        realized_ret = self.y_test.iloc[:, 0].to_numpy()

        df = pd.DataFrame(
            {
                "yyyymm": self.X_test["yyyymm"].to_numpy(),
                "encoded_group": encoded_pred,
                "realized_return": realized_ret,
            }
        )

        monthly_group_means = (
            df.groupby(["yyyymm", "encoded_group"], as_index=False)["realized_return"]
            .mean()
            .rename(columns={"realized_return": "monthly_mean_realized_return"})
        )

        group_avg = (
            monthly_group_means.groupby("encoded_group", as_index=False)["monthly_mean_realized_return"]
            .mean()
            .rename(columns={"monthly_mean_realized_return": "mean_realized_return"})
            .sort_values("encoded_group")
        )

        fig = plot_encoded_group_mean_returns(group_avg)
        return group_avg, fig

    def t_test_long_short(
        self,
        percentage: int = .1,
        alternative: str = "greater",  # "greater" or "two-sided"
        ) -> pd.DataFrame:
        """
        Computes a Fama-French style t-test on ensemble predictions only.

        Steps:
        - Each month: rank stocks using ensemble score
        - Build long (top_k) and short (bottom_k) portfolio
        - Compute monthly long-short returns
        - Run one-sample t-test on mean return

        H0: E[R_long-short] = 0
        """
        top_k = int(percentage * np.min(self.group_test))  # ensure top_k is feasible for all months

        scores = self.ensemble.predict(self.X_test, groups=self.group_test)

        y = self.y_test.iloc[:, 0].to_numpy()  # assumes single reference target
        groups = self.group_test

        long_short_returns = []

        cursor = 0
        for g in groups:
            sl = slice(cursor, cursor + g)

            s = scores[sl]
            r = y[sl]

            k = min(top_k, g // 2)

            order = np.argsort(s)

            short_idx = order[:k]
            long_idx = order[-k:]

            r_long = np.mean(r[long_idx])
            r_short = np.mean(r[short_idx])

            long_short_returns.append(r_long - r_short)

            cursor += g

        R = np.array(long_short_returns)

        # =========================
        # stats
        # =========================
        mean = R.mean()
        std = R.std(ddof=1)
        T = len(R)

        t_stat = mean / (std / np.sqrt(T)) if std > 0 else np.nan

        if alternative == "greater":
            p_value = stats.t.sf(t_stat, df=T - 1)
        else:
            p_value = stats.t.sf(np.abs(t_stat), df=T - 1) * 2

        return pd.DataFrame([{
            "name": "t-test_long_short",
            "mean_return": mean,
            "std_return": std,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_months": T,
            "top_k": top_k,
            "percentage": percentage,
            "model": "ensemble",
            "interpretation": (
                "Significantly positive alpha (reject H0)" if (p_value < 0.05 and mean > 0)
                else "Significantly negative alpha (reject H0)" if (p_value < 0.05 and mean < 0)
                else "No statistically significant alpha (fail to reject H0)"
            ),
            "predictive_power": (
                "YES" if (p_value < 0.05 and mean > 0)
                else "NO"
            )
        }])

    def t_test_long_short_nw(
        self,
        percentage: int = .1,
        lag: int = 3,  # HAC lag (3–6 months typical)
    ) -> pd.DataFrame:

        top_k = int(percentage * np.min(self.group_test))  # ensure top_k is feasible for all months
        scores = self.ensemble.predict(self.X_test, groups=self.group_test)

        y = self.y_test.iloc[:, 0].to_numpy()
        groups = self.group_test

        long_short_returns = []

        cursor = 0
        for g in groups:
            sl = slice(cursor, cursor + g)

            s = scores[sl]
            r = y[sl]

            k = min(top_k, g // 2)

            order = np.argsort(s)

            long_idx = order[-k:]
            short_idx = order[:k]

            r_long = np.mean(r[long_idx])
            r_short = np.mean(r[short_idx])

            long_short_returns.append(r_long - r_short)

            cursor += g

        R = np.array(long_short_returns)

        # =========================
        # NEWEY-WEST t-test
        # =========================
        X = np.ones(len(R))
        model = sm.OLS(R, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})

        mean = R.mean()
        t_stat = model.tvalues[0]
        p_value = model.pvalues[0]

        return pd.DataFrame([{
            "name": f"long_short_nw_lag{lag}",
            "mean_return": mean,
            "std_return": R.std(ddof=1),
            "t_stat": t_stat,
            "p_value": p_value,
            "n_months": len(R),
            "top_k": top_k,
            "percentage": percentage,
            # "hac_lag": lag,
            "model": "ensemble",
            "interpretation": (
                "Significantly positive alpha (reject H0)" if (p_value < 0.05 and mean > 0)
                else "Significantly negative alpha (reject H0)" if (p_value < 0.05 and mean < 0)
                else "No statistically significant alpha (fail to reject H0)"
            ),
            "predictive_power": (
                "YES" if (p_value < 0.05 and mean > 0)
                else "NO"
            )
        }])

    def ndcg_at_k(self, scores, labels, k):
        order = np.argsort(scores)[::-1][:k]
        gains = labels[order]
        discounts = np.log2(np.arange(2, len(gains) + 2))
        dcg = np.sum(gains / discounts)

        ideal_order = np.argsort(labels)[::-1][:k]
        ideal_gains = labels[ideal_order]
        idcg = np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2)))

        return dcg / idcg if idcg > 0 else 0.0

    def precision_at_k(self, scores, labels, k):
        k = min(k, len(scores))
        top_k_pred = set(np.argsort(scores)[::-1][:k])
        top_k_actual = set(np.argsort(labels)[::-1][:k])
        return len(top_k_pred & top_k_actual) / k

    # ------------------------------------------------------------------
    # Drawdown diagnostics
    # ------------------------------------------------------------------

    def diagnose_drawdown(
        self,
        start_yyyymm: int,
        end_yyyymm: int,
        rolling_window: int = 12,
        top_pct: float = 0.1,
    ) -> dict[str, go.Figure]:
        """Four diagnostic plots for a drawdown period.

        Parameters
        ----------
        start_yyyymm / end_yyyymm : int
            YYYYMM integers defining the drawdown window (e.g. 202003, 202006).
            The window is shaded in red on every plot.
        rolling_window : int
            Months used for rolling Spearman and rolling FF5 regressions.
        top_pct : float
            Fraction defining "top" stocks each month (default 0.10 = top decile).

        Returns
        -------
        dict with four Plotly figures keyed by:
            "precision"           – monthly Precision@decile
            "spearman"            – rolling Spearman ρ (score vs realized return)
            "returns_comparison"  – model top-decile vs actual top-decile returns
            "rolling_ff5"         – rolling FF5 factor betas
        """
        from src.portfolio.PortfolioAnalyzer import _load_ff5_factors

        scores = self.ensemble.predict(self.X_test, groups=list(self.group_test))
        y_ref  = self.y_test.iloc[:, 0].to_numpy()
        yyyymm_arr = self.X_test["yyyymm"].to_numpy().astype(int)

        # ── Per-month metrics ──────────────────────────────────────────
        months, precision_vals, spearman_vals = [], [], []
        model_top_ret, actual_top_ret = [], []

        cursor = 0
        for g in self.group_test:
            sl   = slice(cursor, cursor + g)
            month = int(yyyymm_arr[cursor])
            s, r  = scores[sl], y_ref[sl]
            k     = max(1, int(g * top_pct))

            pred_top   = set(np.argsort(s)[::-1][:k])
            actual_top = set(np.argsort(r)[::-1][:k])

            months.append(month)
            precision_vals.append(len(pred_top & actual_top) / k)
            spearman_vals.append(stats.spearmanr(s, r).statistic)
            model_top_ret.append(float(r[np.argsort(s)[::-1][:k]].mean()))
            actual_top_ret.append(float(r[np.argsort(r)[::-1][:k]].mean()))
            cursor += g

        df = pd.DataFrame({
            "yyyymm":        months,
            "date":          pd.to_datetime([str(m) for m in months], format="%Y%m"),
            "precision":     precision_vals,
            "spearman":      spearman_vals,
            "model_top_ret": model_top_ret,
            "actual_top_ret":actual_top_ret,
        })
        df["rolling_spearman"] = df["spearman"].rolling(rolling_window, min_periods=3).mean()

        d0 = pd.to_datetime(str(start_yyyymm), format="%Y%m")
        d1 = pd.to_datetime(str(end_yyyymm),   format="%Y%m")

        def shade(fig):
            fig.add_vrect(x0=d0, x1=d1, fillcolor="red", opacity=0.12, line_width=0)

        # ── Plot 1: Monthly Precision@decile ───────────────────────────
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["date"], y=df["precision"],
            mode="lines", name=f"Precision@{int(top_pct*100)}%",
            line=dict(color="#2196F3"),
        ))
        fig1.add_hline(y=top_pct, line_dash="dash", line_color="grey",
                       annotation_text=f"random baseline ({top_pct:.0%})")
        shade(fig1)
        fig1.update_layout(
            title=f"Monthly Precision@{int(top_pct*100)}% — drawdown {start_yyyymm}→{end_yyyymm}",
            xaxis_title="Month", yaxis_title="Precision",
            yaxis=dict(tickformat=".0%"), template="plotly_white",
        )

        # ── Plot 2: Rolling Spearman correlation ───────────────────────
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["date"], y=df["rolling_spearman"],
            mode="lines", name=f"{rolling_window}m rolling Spearman ρ",
            line=dict(color="#4CAF50"),
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="grey")
        shade(fig2)
        fig2.update_layout(
            title=f"Rolling {rolling_window}m Spearman ρ (score vs realized return)",
            xaxis_title="Month", yaxis_title="Spearman ρ", template="plotly_white",
        )

        # ── Plot 3: Model top-decile return vs actual top-decile return ─
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["date"], y=df["model_top_ret"],
            mode="lines", name="Model top-decile (realized)",
            line=dict(color="#FF5722"),
        ))
        fig3.add_trace(go.Scatter(
            x=df["date"], y=df["actual_top_ret"],
            mode="lines", name="Actual top-decile (oracle)",
            line=dict(color="#9C27B0", dash="dot"),
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="grey")
        shade(fig3)
        fig3.update_layout(
            title="Model top-decile vs oracle top-decile — mean monthly return",
            xaxis_title="Month", yaxis_title="Mean return",
            yaxis=dict(tickformat=".2%"), template="plotly_white",
        )

        # ── Plot 4: Rolling FF5 betas ──────────────────────────────────
        ff5 = _load_ff5_factors()
        ff5 = ff5.rename(columns={"yyyymm": "yyyymm"})

        port = df[["yyyymm", "model_top_ret"]].copy()
        port = port.merge(ff5, on="yyyymm", how="inner")
        port["excess_ret"] = port["model_top_ret"] - port["RF"]
        port["date"] = pd.to_datetime(port["yyyymm"].astype(str), format="%Y%m")

        factors = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
        factor_colors = {"Mkt_RF": "#1976D2", "SMB": "#388E3C",
                         "HML": "#F57C00", "RMW": "#7B1FA2", "CMA": "#D32F2F"}

        n = len(port)
        roll_betas = {f: [np.nan] * n for f in factors}
        roll_alpha = [np.nan] * n

        for i in range(rolling_window - 1, n):
            w = port.iloc[i - rolling_window + 1 : i + 1]
            X_w = sm.add_constant(w[factors].astype(float))
            y_w = w["excess_ret"].astype(float)
            try:
                res = sm.OLS(y_w, X_w).fit()
                roll_alpha[i] = res.params["const"]
                for f in factors:
                    roll_betas[f][i] = res.params[f]
            except Exception:
                pass

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=port["date"], y=roll_alpha,
            mode="lines", name="Alpha",
            line=dict(color="black", dash="dot"),
        ))
        for f in factors:
            fig4.add_trace(go.Scatter(
                x=port["date"], y=roll_betas[f],
                mode="lines", name=f,
                line=dict(color=factor_colors[f]),
            ))
        fig4.add_hline(y=0, line_dash="dash", line_color="grey")
        shade(fig4)
        fig4.update_layout(
            title=f"Rolling {rolling_window}m FF5 betas — model top-decile portfolio",
            xaxis_title="Month", yaxis_title="Coefficient", template="plotly_white",
        )

        # ── Plot 5: Precision distribution (histogram) ────────────────
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(
            x=df["precision"],
            nbinsx=20,
            name="All months",
            marker_color="#2196F3", opacity=0.7,
        ))
        drawdown_mask = (df["yyyymm"] >= start_yyyymm) & (df["yyyymm"] <= end_yyyymm)
        if drawdown_mask.any():
            fig5.add_trace(go.Histogram(
                x=df.loc[drawdown_mask, "precision"],
                nbinsx=20,
                name="Drawdown months",
                marker_color="red", opacity=0.7,
            ))
        fig5.add_vline(
            x=top_pct, line_dash="dash", line_color="grey",
            annotation_text=f"random ({top_pct:.0%})", annotation_position="top right",
        )
        fig5.update_layout(
            barmode="overlay",
            title=f"Distribution of monthly Precision@{int(top_pct*100)}%",
            xaxis_title=f"Precision@{int(top_pct*100)}%",
            yaxis_title="Number of months",
            xaxis=dict(tickformat=".0%"),
            template="plotly_white",
        )

        return {
            "precision":              fig1,
            "spearman":               fig2,
            "returns_comparison":     fig3,
            "rolling_ff5":            fig4,
            "precision_distribution": fig5,
        }
