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
            "all_top_1": self.strategy_all_top_1
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
