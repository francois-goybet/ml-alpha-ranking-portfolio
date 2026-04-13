import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.visualization.portfolio_plots import plot_pnl, plot_drawdown
import statsmodels.api as sm

class PortfolioAnalyzer:

    def __init__(
        self,
        rf_df: pd.DataFrame,
        ret_sp500: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        date_col: str = "yyyymm"
    ):
        self.rf_df = rf_df.copy()
        self.ret_sp500 = ret_sp500.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.date_col = date_col

        self.start = self.X_test[self.date_col].min()
        print("PortfolioAnalyzer initialized.")
        print(self.start)

    def pnl_rf(self):
        df = self.rf_df[self.rf_df["yyyymm"] >= self.start].sort_values("yyyymm")
        wealth = (1 + df["rf"].astype(float).to_numpy()).cumprod()

        fig = plot_pnl(pd.DataFrame({
            "yyyymm": df["yyyymm"].values,
            "wealth": wealth
        }), title="Risk-Free PnL")
        return fig
  
    def pnl_custom_strategy(
        self,
        strategy_df: pd.DataFrame,
        strategy_name: str = "Custom Strategy",
        bps: float = 10
    ):
        """
        Portfolio PnL with stocks + risk-free asset + transaction costs.
        """

        strat = strategy_df.copy()
        cost_rate = bps / 10000  # bps → decimal

        # --- 1. Merge stock returns
        data = strat.merge(
            self.X_test[["yyyymm", "permno", "ret_1m"]],
            on=["yyyymm", "permno"],
            how="left"
        )

        rf_df = self.rf_df[["yyyymm", "rf"]].copy()
        ret_sp500 = self.ret_sp500[["yyyymm", "ret"]].copy()

        rf_df["rf_1m"] = rf_df["rf"].shift(-1)
        ret_sp500["ret_1m_sp500"] = ret_sp500["ret"].shift(-1)

        # --- 2. merge rf
        data = data.merge(
            rf_df[["yyyymm", "rf_1m"]],
            on="yyyymm",
            how="left"
        )

        if data["rf_1m"].isna().any():
            missing_rf = data[data["rf_1m"].isna()]["yyyymm"].unique()
            raise ValueError(f"Missing rf for dates: {missing_rf}")

        # --- 3. replace rf asset return
        data["ret_1m"] = np.where(
            data["permno"] == -1,
            data["rf_1m"],
            data["ret_1m"]
        )

        if data["ret_1m"].isna().any():
            raise ValueError("Missing stock returns")

        # --- 4. gross returns
        data["gross_ret"] = data["weight"] * data["ret_1m"]

        # --- 5. turnover (portfolio-level correct approximation)
        data = data.sort_values(["permno", "yyyymm"])

        data["prev_weight"] = data.groupby("permno")["weight"].shift(1).fillna(0)

        data["turnover"] = (data["weight"] - data["prev_weight"]).abs()

        data["cost"] = data["turnover"] * cost_rate

        # net contribution
        data["net_ret"] = data["gross_ret"] - data["cost"]

        # --- 6. aggregate portfolio
        port = data.groupby("yyyymm", as_index=False)["net_ret"].sum()
        port = port.sort_values("yyyymm")

        # --- 7. wealth
        wealth = 1.0
        wealth_list = []

        for r in port["net_ret"].values:
            wealth *= (1 + r)
            wealth_list.append(wealth)

        port["wealth"] = wealth_list

        df = port[["yyyymm", "wealth"]]

        # --- 8. metrics
        monthly_ret = port["net_ret"].astype(float)
        n_months = len(monthly_ret)

        mean_ret = monthly_ret.mean() if n_months > 0 else np.nan
        std_ret = monthly_ret.std(ddof=1) if n_months > 1 else np.nan

        annualized_sharpe = np.nan
        if std_ret and std_ret > 0:
            annualized_sharpe = (mean_ret / std_ret) * np.sqrt(12)

        annualized_return = np.nan
        if n_months > 0:
            ending = df["wealth"].iloc[-1]
            annualized_return = ending ** (12 / n_months) - 1

        rolling_peak = df["wealth"].cummax()
        drawdown = df["wealth"] / rolling_peak - 1

        df_drawdown = pd.DataFrame({
            "yyyymm": df["yyyymm"],
            "drawdown": drawdown
        })

        max_drawdown = float(drawdown.min())

        wins = monthly_ret[monthly_ret > 0]
        losses = monthly_ret[monthly_ret < 0]

        avg_win = float(wins.mean()) if not wins.empty else np.nan
        avg_loss = float(losses.mean()) if not losses.empty else np.nan

        # --- 9. regression vs SP500
        port["yyyymm"] = port["yyyymm"].astype(int)
        ret_sp500["yyyymm"] = ret_sp500["yyyymm"].astype(int)
        rf_df["yyyymm"] = rf_df["yyyymm"].astype(int)

        merged = port.merge(
            ret_sp500[["yyyymm", "ret_1m_sp500"]],
            on="yyyymm",
            how="left"
        ).merge(
            rf_df[["yyyymm", "rf_1m"]],
            on="yyyymm",
            how="left"
        )

        merged["excess_ret"] = merged["net_ret"] - merged["rf_1m"]
        merged["excess_sp500"] = merged["ret_1m_sp500"] - merged["rf_1m"]
        
        merged["excess_ret"] = merged["excess_ret"].astype(float)
        merged["excess_sp500"] = merged["excess_sp500"].astype(float)
        
        X = sm.add_constant(merged["excess_sp500"])
        y = merged["excess_ret"]

        model = sm.OLS(y, X).fit()

        sp_500_ols_metrics = {
            "alpha": model.params["const"],
            "beta": model.params["excess_sp500"],
            "r2": model.rsquared,
            "adj_r2": model.rsquared_adj,
            "alpha_tstat": model.tvalues["const"],
            "beta_tstat": model.tvalues["excess_sp500"],
            "alpha_pvalue": model.pvalues["const"],
            "beta_pvalue": model.pvalues["excess_sp500"],
        }

        metrics_df = pd.DataFrame([{
            "annualized_sharpe_ratio": annualized_sharpe,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "AvgWin": avg_win,
            "AvgLoss": avg_loss,
            "Total Month": n_months,
        }])

        pnl_fig = plot_pnl(df, title=strategy_name + " PnL")
        drawdown_fig = plot_drawdown(df_drawdown, title=strategy_name + " Drawdown")

        return pnl_fig, drawdown_fig, metrics_df, sp_500_ols_metrics