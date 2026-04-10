"""Ranking model hierarchy: BaseRankingModel → XGBoostRanker / LGBMRanker."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    warnings.warn("LightGBM is not installed. Install it to use LGBMRanker.")
###
# One thing to note is the parameter group needed for both xgboost and lightgbm ranker : 
# basically you have to provide the model the indices corresponding to the same month so it knows what
# group of rows to compare to each other. So the dataframe should be ordered by yyyymm.

class BaseRankingModel(ABC):
    """Abstract base for all learning-to-rank models."""

    def __init__(
        self,
        num_rounds: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbosity: int = 0,
    ):
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.verbosity = verbosity

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "model_")

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        # groups: one integer per query (month), equal to the number of stocks in that month.
        # The ranker compares stocks WITHIN each group only, never across months.
        # Must satisfy: sum(groups) == len(X).
        # Build with: groups = df.groupby(date_col).size().tolist()
        groups: list[int] | np.ndarray,
        eval_set: Optional[tuple] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> BaseRankingModel: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]: ...

    @abstractmethod
    def save_model(self, filepath: str) -> None: ...

    @abstractmethod
    def load_model(self, filepath: str) -> BaseRankingModel: ...

    def get_shap_values(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        explainer = shap.TreeExplainer(self.model_)
        return explainer.shap_values(X)

    def get_params(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if not k.endswith("_") and not k.startswith("_")
        }

    def get_training_history(self) -> Dict[str, Dict[str, list[float]]]:
        return getattr(self, "training_history_", {}).copy()

    def set_params(self, **kwargs) -> BaseRankingModel:
        if self.is_fitted:
            warnings.warn("set_params on a fitted model resets it. Call fit() again.")
            for attr in ("model_", "feature_names_", "training_history_"):
                if hasattr(self, attr):
                    delattr(self, attr)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self


class XGBoostRanker(BaseRankingModel):
    """XGBoost learning-to-rank model.

    Supports rank:pairwise, rank:ndcg, rank:map objectives.

    Key LTR parameters:
        lambdarank_pair_method: "topk" focuses loss on top-K stocks per month (recommended).
            "mean" samples pairs uniformly across all stocks in the month.
        lambdarank_num_pair_per_sample: When pair_method="topk", this is the cutoff K
            (e.g. 10 means optimize NDCG@10 — get the top 10 stocks right each month).
            When pair_method="mean", it is the number of pairs sampled per stock.
        ndcg_exp_gain: Set False when y is continuous returns. Set True only for
            integer relevance grades (web-search style, grades must be <= 31).
        lambdarank_normalization: Normalizes leaf values by lambda gradient. Disable
            if training stagnates.
        lambdarank_score_normalization: Normalizes gradient by score difference between
            pairs. Acts as regularization. Disable if training won't converge.
    """

    def __init__(
        self,
        num_rounds: int = 100,
        objective: str = "rank:ndcg",
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbosity: int = 0,
        # --- LTR-specific ---
        ndcg_exp_gain: bool = False,
        lambdarank_pair_method: str = "topk",
        lambdarank_num_pair_per_sample: int = 10,
        lambdarank_normalization: bool = True,
        lambdarank_score_normalization: bool = True,
        lambdarank_unbiased: bool = False,
        lambdarank_bias_norm: float = 2.0,
    ):
        super().__init__(
            num_rounds=num_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=verbosity,
        )
        self.objective = objective
        self.ndcg_exp_gain = ndcg_exp_gain
        self.lambdarank_pair_method = lambdarank_pair_method
        self.lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample
        self.lambdarank_normalization = lambdarank_normalization
        self.lambdarank_score_normalization = lambdarank_score_normalization
        self.lambdarank_unbiased = lambdarank_unbiased
        self.lambdarank_bias_norm = lambdarank_bias_norm

    def _xgb_params(self) -> dict:
        return {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.random_state,
            "verbosity": self.verbosity,
            "ndcg_exp_gain": self.ndcg_exp_gain,
            "lambdarank_pair_method": self.lambdarank_pair_method,
            "lambdarank_num_pair_per_sample": self.lambdarank_num_pair_per_sample,
            "lambdarank_normalization": self.lambdarank_normalization,
            "lambdarank_score_normalization": self.lambdarank_score_normalization,
            "lambdarank_unbiased": self.lambdarank_unbiased,
            "lambdarank_bias_norm": self.lambdarank_bias_norm,
        }

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        # groups: one integer per month = number of stocks in that month.
        # Ranking loss is computed within each group; sum(groups) must equal len(X).
        groups: list[int] | np.ndarray,
        eval_set: Optional[tuple] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> XGBoostRanker:
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        dtrain = xgb.DMatrix(X_arr, label=y_arr, feature_names=feature_names)
        dtrain.set_group(groups)

        evals = []
        if eval_set is not None and eval_groups is not None:
            X_e, y_e = eval_set
            X_e = X_e.values if isinstance(X_e, pd.DataFrame) else X_e
            y_e = y_e.values if isinstance(y_e, pd.Series) else y_e
            deval = xgb.DMatrix(X_e, label=y_e, feature_names=feature_names)
            deval.set_group(eval_groups)
            evals = [(dtrain, "train"), (deval, "eval")]

        evals_result: Dict[str, Dict[str, list[float]]] = {}
        self.model_ = xgb.train(
            self._xgb_params(),
            dtrain,
            num_boost_round=self.num_rounds,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=verbose,
        )
        self.feature_names_ = feature_names
        self.training_history_ = evals_result
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        iteration_range: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        dtest = xgb.DMatrix(X_arr, feature_names=getattr(self, "feature_names_", None))
        kwargs = {"iteration_range": iteration_range} if iteration_range else {}
        return self.model_.predict(dtest, **kwargs)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        return self.model_.get_score(importance_type=importance_type)

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        self.model_.save_model(filepath)

    def load_model(self, filepath: str) -> XGBoostRanker:
        self.model_ = xgb.Booster()
        self.model_.load_model(filepath)
        return self


# ---------------------------------------------------------------------------
# Label encoders for LightGBM (requires integer grades per month)
#
# Input data is expected as a flat DataFrame with columns:
#   permno  (stock id), yyyymm (period), ret (return), f1, f2, ... (features)
# groups must be built AFTER encoding so row order is preserved:
#   groups = df.groupby("yyyymm").size().tolist()
# ---------------------------------------------------------------------------

def encode_labels_quintile(y: pd.Series, groups: list[int]) -> np.ndarray:
    """
    Encode continuous returns into quintile grades {0, 1, 2, 3, 4} computed
    independently within each month (group).

    Grade 4 = top 20% of returns that month, grade 0 = bottom 20%.
    Ties are handled by assigning the lower grade (duplicates='drop' falls back
    to fewer bins when a month has too few unique values).

    Args:
        y: Flat Series of continuous returns, same row order as X.
        groups: Stock counts per month; sum(groups) must equal len(y).

    Returns:
        Integer ndarray of grades, same length as y.
    """
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    out = np.empty(len(y_arr), dtype=np.int64)
    cursor = 0
    for g in groups:
        sl = slice(cursor, cursor + g)
        chunk = pd.Series(y_arr[sl])
        out[sl] = pd.qcut(chunk, q=5, labels=False, duplicates="drop").astype(np.int64)
        cursor += g
    return out


def encode_labels_decile(y: pd.Series, groups: list[int]) -> np.ndarray:
    """
    Same as encode_labels_quintile but uses 10 bins (deciles).
    Grade 9 = top 10%, grade 0 = bottom 10%.
    """
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    out = np.empty(len(y_arr), dtype=np.int64)
    cursor = 0
    for g in groups:
        sl = slice(cursor, cursor + g)
        chunk = pd.Series(y_arr[sl])
        out[sl] = pd.qcut(chunk, q=10, labels=False, duplicates="drop").astype(np.int64)
        cursor += g
    return out


def encode_labels_binary(y: pd.Series, groups: list[int]) -> np.ndarray:
    """
    Binary encoding: 1 if return >= median of the month, 0 otherwise.
    Useful for simple top-half vs bottom-half ranking.
    """
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    out = np.empty(len(y_arr), dtype=np.int64)
    cursor = 0
    for g in groups:
        sl = slice(cursor, cursor + g)
        chunk = y_arr[sl]
        out[sl] = (chunk >= np.median(chunk)).astype(np.int64)
        cursor += g
    return out


# Registry so callers can select an encoder by name from config
_LABEL_ENCODERS = {
    "quintile": encode_labels_quintile,
    "decile": encode_labels_decile,
    "binary": encode_labels_binary,
}


class LGBMRanker(BaseRankingModel):
    """LightGBM learning-to-rank model."""

    def __init__(
        self,
        num_rounds: int = 100,
        objective: str = "lambdarank",
        learning_rate: float = 0.1,
        max_depth: int = 6,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbosity: int = -1,
        # --- LTR-specific ---
        lambdarank_truncation_level: int = 10,
        label_gain: Optional[list[float]] = None,
        # label_encoder: how to convert continuous returns to integer grades.
        # Built-in options: "quintile" (default), "decile", "binary".
        # Pass a callable f(y, groups) -> np.ndarray for a custom scheme.
        # Set to None to skip encoding (y must already be integer grades).
        label_encoder: str | callable | None = "quintile",
    ):
        super().__init__(
            num_rounds=num_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=verbosity,
        )
        self.objective = objective
        self.num_leaves = num_leaves
        self.lambdarank_truncation_level = lambdarank_truncation_level
        # label_gain maps integer relevance grades to NDCG gain values
        # default None lets LightGBM use its built-in [0,1,3,7,15,...]
        self.label_gain = label_gain
        self.label_encoder = label_encoder

    def _lgb_params(self) -> dict:
        params = {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "bagging_fraction": self.subsample,
            "feature_fraction": self.colsample_bytree,
            "seed": self.random_state,
            "verbosity": self.verbosity,
            "lambdarank_truncation_level": self.lambdarank_truncation_level,
        }
        if self.label_gain is not None:
            params["label_gain"] = self.label_gain
        return params

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        # groups: one integer per month = number of stocks in that month.
        # Ranking loss is computed within each group; sum(groups) must equal len(X).
        groups: list[int] | np.ndarray,
        eval_set: Optional[tuple] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> LGBMRanker:
        if lgb is None:
            raise ImportError("LightGBM is not installed.")
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_s = y if isinstance(y, pd.Series) else pd.Series(y)

        # --- Encode continuous returns to integer grades if needed ---
        # LightGBM ranking requires integer labels (higher = more relevant).
        # label_encoder handles this automatically; set to None if y is already integers.
        if self.label_encoder is not None:
            if callable(self.label_encoder) and not isinstance(self.label_encoder, str):
                encoder_fn = self.label_encoder
            elif self.label_encoder in _LABEL_ENCODERS:
                encoder_fn = _LABEL_ENCODERS[self.label_encoder]
            else:
                raise ValueError(
                    f"Unknown label_encoder '{self.label_encoder}'. "
                    f"Choose from {list(_LABEL_ENCODERS)} or pass a callable."
                )
            y_arr = encoder_fn(y_s, list(groups))
        else:
            y_arr = y_s.values
            if not np.issubdtype(y_arr.dtype, np.integer):
                warnings.warn(
                    "label_encoder=None but y is not integer. "
                    "LGBMRanker requires integer grades. "
                    "Set label_encoder='quintile' or convert y manually.",
                    UserWarning,
                    stacklevel=2,
                )
                y_arr = y_arr.astype(np.int64)

        dtrain = lgb.Dataset(X_arr, label=y_arr, group=groups, feature_name=feature_names or "auto")

        valid_sets = [dtrain]
        valid_names = ["train"]
        if eval_set is not None and eval_groups is not None:
            X_e, y_e = eval_set
            X_e = X_e.values if isinstance(X_e, pd.DataFrame) else X_e
            y_e = y_e.values if isinstance(y_e, pd.Series) else y_e
            deval = lgb.Dataset(X_e, label=y_e, group=eval_groups, reference=dtrain)
            valid_sets.append(deval)
            valid_names.append("eval")

        evals_result: dict = {}
        callbacks = [lgb.record_evaluation(evals_result)]
        if not verbose:
            callbacks.append(lgb.log_evaluation(period=-1))

        self.model_ = lgb.train(
            self._lgb_params(),
            dtrain,
            num_boost_round=self.num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.feature_names_ = feature_names
        self.training_history_ = evals_result
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict(X_arr)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        names = self.model_.feature_name()
        scores = self.model_.feature_importance(importance_type=importance_type)
        return dict(zip(names, scores.tolist()))

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        self.model_.save_model(filepath)

    def load_model(self, filepath: str) -> LGBMRanker:
        if lgb is None:
            raise ImportError("LightGBM is not installed.")
        self.model_ = lgb.Booster(model_file=filepath)
        return self


# ---------------------------------------------------------------------------
# Multi-horizon ranker
# ---------------------------------------------------------------------------

class MultiHorizonRanker:
    """
    Trains one independent ranking model per target horizon.

    Each horizon (e.g. t+1, t+3, t+6) gets its own model because the feature
    importance and optimal hyperparameters often differ across horizons.
    All models share the same backend (xgboost or lightgbm) and hyperparameters.

    Args:
        targets: List of target column names, one per horizon.
            e.g. ["return_t1", "return_t3", "return_t6"]
        backend: "xgboost" or "lightgbm".
        **model_kwargs: Passed directly to XGBoostRanker or LGBMRanker.

    After fit(), access individual models via .models_[target].
    """

    def __init__(
        self,
        targets: list[str],
        backend: str = "xgboost",
        **model_kwargs,
    ):
        if not targets:
            raise ValueError("targets must be a non-empty list of column names.")
        if backend not in ("xgboost", "lightgbm"):
            raise ValueError("backend must be 'xgboost' or 'lightgbm'.")
        self.targets = targets
        self.backend = backend
        self.model_kwargs = model_kwargs

    def _make_model(self) -> BaseRankingModel:
        if self.backend == "xgboost":
            return XGBoostRanker(**self.model_kwargs)
        if self.backend == "lightgbm":
            return LGBMRanker(**self.model_kwargs)

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "models_")

    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        # groups: one integer per month = number of stocks in that month.
        # All target columns share the same group structure (same rows, same dates).
        # sum(groups) must equal len(X).
        groups: list[int] | np.ndarray,
        eval_set: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> MultiHorizonRanker:
        """
        Args:
            X: Feature matrix (rows = stocks × months).
            Y: DataFrame with one column per target horizon.
               e.g. columns ["return_t1", "return_t3", "return_t6"].
            groups: Stock counts per month, shared across all horizons.
            eval_set: Optional (X_val, Y_val) for monitoring.
            eval_groups: Group sizes for the validation set.
        """
        missing = [t for t in self.targets if t not in Y.columns]
        if missing:
            raise ValueError(f"Target columns not found in Y: {missing}")

        self.models_: Dict[str, BaseRankingModel] = {}
        for target in self.targets:
            model = self._make_model()
            es = (eval_set[0], eval_set[1][target]) if eval_set is not None else None
            model.fit(
                X,
                Y[target],
                groups=groups,
                eval_set=es,
                eval_groups=eval_groups,
                verbose=verbose,
            )
            self.models_[target] = model
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Returns a dict {target: score_array} for each horizon."""
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        return {target: model.predict(X) for target, model in self.models_.items()}

    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> Dict[str, Dict[str, float]]:
        """Returns {target: {feature: importance}} for each horizon."""
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        return {
            target: model.get_feature_importance(importance_type)
            for target, model in self.models_.items()
        }


# ---------------------------------------------------------------------------
# Horizon ensemble — combines scores across horizons
# ---------------------------------------------------------------------------

class HorizonEnsemble:
    """
    Aggregates predictions from a fitted MultiHorizonRanker into a single
    ranking score per stock per month.

    Two combination modes:
        "mean_score": average the raw model scores across horizons, then rank.
            Simple and fast. Works well when all horizons are on a comparable
            scale (which they are when using the same ranker objective).
        "mean_rank": convert each horizon's scores to per-month ranks first,
            then average the ranks. More robust to one horizon dominating
            because of score magnitude differences. This is the Borda count
            approach and is common in ensemble ranking literature.

    Args:
        multi_ranker: A fitted MultiHorizonRanker.
        combination: "mean_score" or "mean_rank".
        weights: Optional list of floats, one per target in multi_ranker.targets.
            Allows giving more weight to near-term horizons (e.g. [0.5, 0.3, 0.2]
            for t+1, t+3, t+6). Defaults to equal weights.
    """

    def __init__(
        self,
        multi_ranker: MultiHorizonRanker,
        combination: str = "mean_rank",
        weights: Optional[list[float]] = None,
    ):
        if combination not in ("mean_score", "mean_rank"):
            raise ValueError("combination must be 'mean_score' or 'mean_rank'.")
        if weights is not None and len(weights) != len(multi_ranker.targets):
            raise ValueError("weights must have one value per target.")
        self.multi_ranker = multi_ranker
        self.combination = combination
        self.weights = weights

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        groups: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Returns a single 1-D score array (higher = better rank).

        Args:
            X: Feature matrix.
            groups: Stock counts per month. Required for mean_rank so ranks
                are computed within each month, not globally. If None and
                combination is mean_rank, global ranks are used as a fallback.
        """
        scores_per_horizon = self.multi_ranker.predict(X)  # {target: array}
        targets = self.multi_ranker.targets
        w = self.weights if self.weights is not None else [1.0] * len(targets)
        w_arr = np.array(w, dtype=float)
        w_arr /= w_arr.sum()  # normalise

        if self.combination == "mean_score":
            stacked = np.stack(
                [scores_per_horizon[t] for t in targets], axis=1
            )  # shape (n_stocks, n_horizons)
            return np.average(stacked, axis=1, weights=w_arr)

        # mean_rank: rank within each month, then average
        n = len(next(iter(scores_per_horizon.values())))
        all_ranks = np.zeros((n, len(targets)), dtype=float)
        for col_idx, target in enumerate(targets):
            raw = scores_per_horizon[target]
            if groups is not None:
                # rank within each month separately
                ranked = np.empty_like(raw)
                cursor = 0
                for g in groups:
                    sl = slice(cursor, cursor + g)
                    # higher score → lower rank number → invert for "higher=better"
                    ranked[sl] = pd.Series(raw[sl]).rank(ascending=True).values
                    cursor += g
            else:
                ranked = pd.Series(raw).rank(ascending=True).values
            all_ranks[:, col_idx] = ranked
        return np.average(all_ranks, axis=1, weights=w_arr)
