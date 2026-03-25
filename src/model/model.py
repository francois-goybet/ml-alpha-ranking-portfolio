"""AlphaXGBoost: XGBoost-based learning-to-rank model for stock ranking."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


class AlphaXGBoost:
    """
    Wrapper around XGBoost for learning-to-rank (LTR) stock ranking.

    Uses pairwise loss (rank:pairwise) to predict stock rankings within groups
    (e.g., monthly rankings based on expected returns).

    Attributes:
        model: Underlying XGBoost Booster object
        feature_names: List of feature column names used for training
        params: Dictionary of XGBoost parameters
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AlphaXGBoost model with pairwise ranking objective.

        Args:
            objective: Loss function (default: rank:pairwise for LTR)
            num_rounds: Number of boosting rounds
            learning_rate: Learning rate (eta)
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBoost parameters
        """
        self.config = config
        self.num_rounds = self.config.get("num_rounds", 100)
        self.feature_names: Optional[list[str]] = None
        self.model: Optional[xgb.Booster] = None
        self.training_history: Dict[str, Dict[str, list[float]]] = {}

        # Build parameter dictionary
        self.params = {
            "objective": self.config.get("objective", "rank:pairwise"),
            "learning_rate": self.config.get("learning_rate", 0.1),
            "max_depth": self.config.get("max_depth", 6),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "random_state": self.config.get("random_state", 42),
            "ndcg_exp_gain": self.config.get("ndcg_exp_gain", False),
            "verbosity": self.config.get("verbosity", 0),
        }
        self.params.update(self.config.get("params", {}))

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: list[int] | np.ndarray,
        eval_set: Optional[tuple] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = True,
    ) -> AlphaXGBoost:
        """
        Train the model on ranking data with groups.

        Args:
            X: Feature matrix (DataFrame or ndarray)
            y: Target values (monthly returns for ranking)
            groups: List of group sizes where each element is the number of samples
                   in each group (e.g., [50, 45, 52] for 3 months with 50, 45, 52 stocks)
            eval_set: Optional tuple of (X_eval, y_eval) for early stopping
            eval_groups: Optional group sizes for eval set
            verbose: Whether to print training progress

        Returns:
            self for method chaining
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_data = X.values
        else:
            X_data = X

        # Convert y to numpy if needed
        if isinstance(y, pd.Series):
            y_data = y.values
        else:
            y_data = y

        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_data, label=y_data)
        dtrain.set_group(groups)

        # Setup evaluation set if provided
        evals = []
        if eval_set is not None and eval_groups is not None:
            X_eval, y_eval = eval_set
            if isinstance(X_eval, pd.DataFrame):
                X_eval = X_eval.values
            if isinstance(y_eval, pd.Series):
                y_eval = y_eval.values
            deval = xgb.DMatrix(X_eval, label=y_eval)
            deval.set_group(eval_groups)
            evals = [(dtrain, "train"), (deval, "eval")]

        # Train model
        evals_result: Dict[str, Dict[str, list[float]]] = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_rounds,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=verbose,
        )
        self.training_history = evals_result

        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        iteration_range: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Predict ranking scores for samples.

        Args:
            X: Feature matrix
            iteration_range: Optional tuple (start, end) for using subset of trees

        Returns:
            Array of prediction scores (higher = more likely to outperform)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = X

        dtest = xgb.DMatrix(X_data)
        
        if iteration_range is not None:
            return self.model.predict(dtest, iteration_range=iteration_range)
        else:
            return self.model.predict(dtest)

    def get_feature_importance(self, importance_type: str = "weight") -> dict:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained first.")

        importance = self.model.get_score(importance_type=importance_type)

        if self.feature_names:
            return {self.feature_names[int(fid.split("f")[1])]: v for fid, v in importance.items()}
        return importance

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving.")
        self.model.save_model(filepath)

    def load_model(self, filepath: str) -> AlphaXGBoost:
        """
        Load a previously trained model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            self for method chaining
        """
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        return self

    def get_params(self) -> dict:
        """Get model parameters."""
        return self.params.copy()

    def get_training_history(self) -> Dict[str, Dict[str, list[float]]]:
        """Get training history captured during fit (train/eval metrics per iteration)."""
        return self.training_history.copy()

    def set_params(self, **kwargs) -> AlphaXGBoost:
        """
        Set model hyperparameters.

        Args:
            **kwargs: Parameter names and values to update

        Returns:
            self for method chaining
        """
        self.params.update(kwargs)
        return self
