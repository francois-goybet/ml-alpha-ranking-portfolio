import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.visualization.portfolio_plots import plot_pnl


class PortfolioAnalyzer:

    def __init__(
        self,
        rf_df: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        date_col: str = "yyyymm"
    ):
        self.rf_df = rf_df.copy()
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
  
    def pnl_custom_strategy(self, strategy_df: pd.DataFrame):
        """
        Portfolio PnL with stocks + risk-free asset.

        strategy_df:
            columns = ["yyyymm", "permno", "weight"]
            permno = -1 means risk-free asset

        df_rf:
            columns = ["yyyymm", "rf"]

        Returns:
            DataFrame: ["yyyymm", "wealth"]
        """

        import pandas as pd
        import numpy as np

        strat = strategy_df.copy()

        # --- 1. Merge stock returns
        data = strat.merge(
            self.X_test[["yyyymm", "permno", "ret"]],
            on=["yyyymm", "permno"],
            how="left"
        )

        # --- 2. Merge risk-free rate
        data = data.merge(
            self.rf_df[["yyyymm", "rf"]],
            on="yyyymm",
            how="left"
        )

        # --- 3. Check missing rf
        if data["rf"].isna().any():
            missing_rf = data[data["rf"].isna()]["yyyymm"].unique()
            raise ValueError(f"Missing rf for dates: {missing_rf}")

        # --- 4. Replace return by rf for risk-free asset
        data["ret"] = np.where(data["permno"] == -1, data["rf"], data["ret"])

        # --- 5. Check missing stock returns
        if data["ret"].isna().any():
            missing = data[data["ret"].isna()]
            raise ValueError(f"Missing stock returns:\n{missing.head()}")

        # --- 6. Compute weighted returns
        data["weighted_ret"] = data["weight"] * data["ret"]

        # --- 7. Aggregate by date
        port = data.groupby("yyyymm", as_index=False)["weighted_ret"].sum()
        port = port.sort_values("yyyymm")

        # --- 8. Wealth evolution
        wealth = 1.0
        wealth_list = []

        for r in port["weighted_ret"].values:
            wealth *= (1 + r)
            wealth_list.append(wealth)

        port["wealth"] = wealth_list

        df = port[["yyyymm", "wealth"]]
        return plot_pnl(df, title="Custom Strategy PnL") 