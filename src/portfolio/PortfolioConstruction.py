import pandas as pd


class PortfolioConstruction:
    def __init__(
        self,
        rf_df: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
    ):
        self.rf_df = rf_df.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.y_pred = y_pred.copy()
        self.strategies = {
            "rf_only": self.strategy_rf_only,
            "all_top_1": self.strategy_all_top_1,
            "all_top_10_equal_weights": self.strategy_all_top_10_equal_weights,
            "top_10_market_cap": self.strategy_top_10_market_cap,
            "top_20_market_cap": lambda: self.strategy_top_N_market_cap(N=20),
            "top_50_market_cap": lambda: self.strategy_top_N_market_cap(N=50),
            "top_100_market_cap": lambda: self.strategy_top_N_market_cap(N=100),
        }

    def strategy_rf_only(self) -> pd.DataFrame:
        """
        For each month, invest 100% in the risk-free asset (permno=-1).
        """
        rf_strategy = self.rf_df[["yyyymm"]].copy()
        # keep months that are in the test set
        rf_strategy = rf_strategy[rf_strategy["yyyymm"] >= self.X_test["yyyymm"].min()]
        rf_strategy = rf_strategy[rf_strategy["yyyymm"] <= self.X_test["yyyymm"].max()]
        
        rf_strategy["permno"] = -1
        rf_strategy["weight"] = 1.0
        return rf_strategy[["yyyymm", "permno", "weight"]]
    
    def strategy_all_top_1(self) -> pd.DataFrame:
        """
        For each month, select the top-1 stock by y_pred score.
        Add risk-free asset (permno=-1) with weight 0.
        """
        base = self.X_test[["yyyymm", "permno"]].copy()
        base["score"] = self.y_pred
        base = base[base["permno"] != -1]
        
        idx_top = base.groupby("yyyymm")["score"].idxmax()
        top = base.loc[idx_top, ["yyyymm", "permno"]].copy()
        top["weight"] = 1.0
        
        rf = top[["yyyymm"]].copy()
        rf["permno"] = -1
        rf["weight"] = 0.0
        
        strategy_df = pd.concat([top, rf], ignore_index=True)
        return strategy_df[["yyyymm", "permno", "weight"]].sort_values(
            ["yyyymm", "weight"], ascending=[True, False]
        ).reset_index(drop=True)

    def strategy_all_top_10_equal_weights(self) -> pd.DataFrame:
        """
        For each month, select the top-10 stocks by y_pred score.
        Assign equal weight to each stock.
        No risk-free asset included.
        """
        base = self.X_test[["yyyymm", "permno"]].copy()
        base["score"] = self.y_pred

        # enlever le risk-free si jamais présent
        base = base[base["permno"] != -1]

        # sélectionner top 10 par mois
        top10 = (
            base.sort_values(["yyyymm", "score"], ascending=[True, False])
                .groupby("yyyymm")
                .head(10)
                .copy()
        )

        # equal weight
        top10["weight"] = 1.0 / 10.0

        return top10[["yyyymm", "permno", "weight"]].sort_values(
            ["yyyymm", "permno"]
        ).reset_index(drop=True)
        
    def strategy_top_10_market_cap(self) -> pd.DataFrame:
        """
        For each month:
        - Select top 10 stocks by y_pred score
        - Assign weights proportional to market cap (market_cap_musd)
        - Normalize weights to sum to 1 per month
        """
        base = self.X_test[["yyyymm", "permno", "market_cap_musd"]].copy()
        base["score"] = self.y_pred

        # remove risk-free if present
        base = base[base["permno"] != -1]

        # select top 10 per month
        top10 = (
            base.sort_values(["yyyymm", "score"], ascending=[True, False])
                .groupby("yyyymm")
                .head(10)
                .copy()
        )

        # market cap weighting inside each month
        def normalize_weights(df):
            df = df.copy()
            df["weight"] = df["market_cap_musd"] / df["market_cap_musd"].sum()
            return df

        top10 = top10.groupby("yyyymm", group_keys=False).apply(normalize_weights)

        return top10[["yyyymm", "permno", "weight"]].sort_values(
            ["yyyymm", "permno"]
        ).reset_index(drop=True)
    
    def strategy_top_N_market_cap(self, N: int = 10) -> pd.DataFrame:
        """
        For each month:
        - Select top N stocks by y_pred score
        - Assign weights proportional to market cap
        - Normalize weights to sum to 1 per month
        """

        base = self.X_test[["yyyymm", "permno", "market_cap_musd"]].copy()
        base["score"] = self.y_pred

        # remove risk-free
        base = base[base["permno"] != -1]

        # select top N per month
        topN = (
            base.sort_values(["yyyymm", "score"], ascending=[True, False])
                .groupby("yyyymm")
                .head(N)
                .copy()
        )

        # market cap weighting per month
        def normalize_weights(df):
            df = df.copy()
            df["weight"] = df["market_cap_musd"] / df["market_cap_musd"].sum()
            return df

        topN = topN.groupby("yyyymm", group_keys=False).apply(normalize_weights)

        return topN[["yyyymm", "permno", "weight"]].sort_values(
            ["yyyymm", "permno"]
        ).reset_index(drop=True)