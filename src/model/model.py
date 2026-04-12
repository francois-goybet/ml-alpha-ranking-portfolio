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

# ---------------------------------------------------------------------------
# Custom eval: Rank Precision (top-decile intersection rate) — matches paper definition.
# For each group (month), K = group_size // 10 (top 10% of stocks).
# Rank Precision = |predicted top-decile ∩ actual top-decile| / K
# Random baseline = 10%; good models reach 15–30% (consistent with paper Table 5).
# ---------------------------------------------------------------------------

_META = {"permno", "yyyymm", "ret", "market_cap_musd", "ret_1m", "ret_3m", "ret_6m"}

def _rank_precision_xgb(groups_list: list[list[int]]):
    """XGBoost custom eval: rank precision at top decile.
    groups_list[i] contains group sizes for the i-th eval dataset.
    XGBoost calls feval once per dataset in order (train, eval, ...).
    """
    call_count = [0]

    def eval_fn(preds: np.ndarray, dmat: "xgb.DMatrix"):
        grps = groups_list[call_count[0] % len(groups_list)]
        call_count[0] += 1
        labels = dmat.get_label()
        precisions = []
        cursor = 0
        for g in grps:
            g = int(g)
            sl = slice(cursor, cursor + g)
            top_k = max(1, g // 10)
            pred_top = set(np.argsort(preds[sl])[::-1][:top_k])
            label_top = set(np.argsort(labels[sl])[::-1][:top_k])
            precisions.append(len(pred_top & label_top) / top_k)
            cursor += g
        return "rank_prec@decile", float(np.mean(precisions))
    return eval_fn


def _rank_precision_lgb():
    """LightGBM custom eval: rank precision at top decile."""
    def eval_fn(preds: np.ndarray, dataset: "lgb.Dataset"):
        labels = dataset.get_label()
        groups = dataset.get_group()
        precisions = []
        cursor = 0
        for g in groups:
            g = int(g)
            sl = slice(cursor, cursor + g)
            top_k = max(1, g // 10)
            pred_top = set(np.argsort(preds[sl])[::-1][:top_k])
            label_top = set(np.argsort(labels[sl])[::-1][:top_k])
            precisions.append(len(pred_top & label_top) / top_k)
            cursor += g
        return "rank_prec@decile", float(np.mean(precisions)), True
    return eval_fn


class BaseRankingModel(ABC):
    """Shared interface and hyper-parameter store for all rankers."""

    def __init__(
        self,
        num_rounds: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbosity: int = 0,
        eval_at: list[int] | None = None,
    ) -> None:
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.verbosity = verbosity
        self.eval_at = eval_at if eval_at is not None else [10, 20]

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "model_")

    @abstractmethod
    def fit(self, X, y, groups, **kwargs): ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    @abstractmethod
    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]: ...

    @abstractmethod
    def save_model(self, filepath: str) -> None: ...

    @abstractmethod
    def load_model(self, filepath: str): ...


class XGBoostRanker(BaseRankingModel):
    """XGBoost learning-to-rank model.

    Uses ``xgb.train`` with the ``rank:pairwise`` (or ``rank:ndcg``) objective.
    Groups are passed via the ``qid`` column of a ``DMatrix`` built from the
    cumulative sum of ``groups``.
    """

    def __init__(
        self,
        num_rounds: int = 100,
        objective: str = "rank:pairwise",
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbosity: int = 0,
        ndcg_exp_gain: bool = False,
        eval_at: list[int] | None = None,
        # label_encoder: convert continuous returns to integer grades per month.
        # Built-in options: "quintile", "decile", "binary", "argsort" (default).
        # Pass a callable f(y, groups) -> np.ndarray for a custom scheme.
        # Set to None to skip encoding (y must already be non-negative integers).
        label_encoder: str | callable | None = "argsort",
    ) -> None:
        super().__init__(
            num_rounds=num_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=verbosity,
            eval_at=eval_at,
        )
        self.objective = objective
        self.ndcg_exp_gain = ndcg_exp_gain
        self.label_encoder = label_encoder

    def _xgb_params(self) -> dict:
        params = {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.random_state,
            "verbosity": self.verbosity,
            "eval_metric": [f"ndcg@{k}" for k in self.eval_at] + ["auc"],
            "ndcg_exp_gain": self.ndcg_exp_gain,
        }
        return params

    @staticmethod
    def _build_dmatrix(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: list[int] | np.ndarray,
    ) -> xgb.DMatrix:
        """Build an XGBoost DMatrix with per-row query group ids."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        # qid: one integer per row identifying the group (month)
        qid = np.repeat(np.arange(len(groups)), groups).astype(np.uint32)
        dm = xgb.DMatrix(X_arr, label=y_arr)
        dm.set_uint_info("group", np.array(groups, dtype=np.uint32))
        if isinstance(X, pd.DataFrame):
            dm.feature_names = X.columns.tolist()
        return dm

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: list[int] | np.ndarray,
        eval_set: Optional[tuple] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> "XGBoostRanker":
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        y_s = y if isinstance(y, pd.Series) else pd.Series(y)

  
        # Encode continuous returns to integer grades if needed
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
            y_encoded = encoder_fn(y_s, list(groups))
        else:
            y_encoded = y_s.values

        dtrain = self._build_dmatrix(X, y_encoded, groups)

        evals = [(dtrain, "train")]
        if eval_set is not None and eval_groups is not None:
            X_e, y_e = eval_set
            y_e_s = y_e if isinstance(y_e, pd.Series) else pd.Series(y_e)
            if self.label_encoder is not None:
                y_e_encoded = encoder_fn(y_e_s, list(eval_groups))
            else:
                y_e_encoded = y_e_s.values
            deval = self._build_dmatrix(X_e, y_e_encoded, eval_groups)
            evals.append((deval, "eval"))

        groups_list = [list(groups)]
        if eval_set is not None and eval_groups is not None:
            groups_list.append(list(eval_groups))

        evals_result: dict = {}
        self.model_ = xgb.train(
            self._xgb_params(),
            dtrain,
            num_boost_round=self.num_rounds,
            evals=evals,
            custom_metric=_rank_precision_xgb(groups_list),
            evals_result=evals_result,
            verbose_eval=verbose,
        )
        self.feature_names_ = feature_names
        self.training_history_ = evals_result
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        dm = xgb.DMatrix(X_arr)
        if self.feature_names_ is not None:
            dm.feature_names = self.feature_names_
        return self.model_.predict(dm)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        scores = self.model_.get_score(importance_type=importance_type)
        return scores

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        self.model_.save_model(filepath)

    def load_model(self, filepath: str) -> XGBoostRanker:
        self.model_ = xgb.Booster()
        self.model_.load_model(filepath)
        return self


# ---------------------------------------------------------------------------
# Label encoders — shared by XGBoostRanker and LGBMRanker
#
# Each encoder takes a flat Series y and the group sizes list, and returns
# a np.ndarray of non-negative integers (higher = more relevant).
# groups must be built AFTER encoding so row order is preserved:
#   groups = df.groupby("yyyymm").size().tolist()
# ---------------------------------------------------------------------------

def encode_labels_quintile(y: pd.Series, groups: list[int]) -> np.ndarray:
    """Assign per-month quintile ranks 0-4 (4 = highest return)."""
    out = np.empty(len(y), dtype=np.int64)
    cursor = 0
    for g in groups:
        sl = slice(cursor, cursor + g)
        out[sl] = (
            pd.qcut(y.iloc[sl], q=5, labels=False, duplicates="drop")
            .fillna(0)
            .astype(np.int64)
        )
        cursor += g
    return out


def encode_labels_decile(y: pd.Series, groups: list[int]) -> np.ndarray:
    """Assign per-month decile ranks 0-9 (9 = highest return)."""
    out = np.empty(len(y), dtype=np.int64)
    cursor = 0
    for g in groups:
        sl = slice(cursor, cursor + g)
        out[sl] = (
            pd.qcut(y.iloc[sl], q=10, labels=False, duplicates="drop")
            .fillna(0)
            .astype(np.int64)
        )
        cursor += g
    return out


def encode_labels_binary(y: pd.Series, groups: list[int]) -> np.ndarray:
    """Assign 1 to the top half, 0 to the bottom half, per month."""
    out = np.empty(len(y), dtype=np.int64)
    cursor = 0
    for g in groups:
        sl = slice(cursor, cursor + g)
        median = y.iloc[sl].median()
        out[sl] = (y.iloc[sl] >= median).astype(np.int64)
        cursor += g
    return out

def encode_labels_argsort(y: pd.Series, groups: list[int]) -> np.ndarray:
    """
    Assign per-group ranks using argsort.
    0 = worst, higher = better.
    """
    out = np.empty(len(y), dtype=np.int64)
    cursor = 0

    for g in groups:
        sl = slice(cursor, cursor + g)
        y_group = y.iloc[sl].to_numpy()

        # argsort twice = rank
        ranks = np.argsort(np.argsort(y_group))

        out[sl] = ranks
        cursor += g

    return out


def encode_labels_long_ranker(y: pd.Series, groups: list[int]) -> np.ndarray:
    """Dual-Ranker long-leg labeling: top-decile NDCG emphasis.

    For each month:
      - top_k = max(1, group_size // 10)   (top 10%)
      - Best stock → label top_k, second best → top_k-1, ..., top_k-th → 1
      - All remaining stocks → label 0

    This creates a sparse relevance vector that focuses NDCG loss entirely on
    the top decile while zeroing out mid-ranked noise.

    Note for LightGBM (lambdarank): set ``label_gain`` in the model config to
    cover values 0..top_k, e.g. ``list(range(max_group // 10 + 1))``, otherwise
    LightGBM clips gains at label 4 by default.
    """
    out = np.zeros(len(y), dtype=np.int64)
    cursor = 0
    for g in groups:
        top_k = max(1, g // 10)
        y_group = y.iloc[cursor : cursor + g].to_numpy()
        # indices sorted highest return first
        sorted_idx = np.argsort(y_group)[::-1]
        for rank, idx in enumerate(sorted_idx[:top_k]):
            out[cursor + idx] = top_k - rank  # top_k, top_k-1, ..., 1
        cursor += g
    return out


# Registry so callers can select an encoder by name from config
_LABEL_ENCODERS = {
    "quintile":     encode_labels_quintile,
    "decile":       encode_labels_decile,
    "binary":       encode_labels_binary,
    "argsort":      encode_labels_argsort,
    "long_ranker":  encode_labels_long_ranker,
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
        eval_at: list[int] | None = None,
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
            eval_at=eval_at,
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
            "eval_at": self.eval_at,
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
            y_e_s = y_e if isinstance(y_e, pd.Series) else pd.Series(y_e)
            if self.label_encoder is not None:
                y_e_arr = encoder_fn(y_e_s, list(eval_groups))
            else:
                y_e_arr = y_e_s.to_numpy(dtype=np.float32, na_value=0)
            deval = lgb.Dataset(X_e, label=y_e_arr, group=eval_groups, reference=dtrain)
            valid_sets.append(deval)
            valid_names.append("eval")

        evals_result: dict = {}
        callbacks = [lgb.record_evaluation(evals_result)]
        if verbose:
            def _fmt_callback(env):
                row_parts = [f"[{env.iteration}]"]
                for ds_name, metric, value, _ in sorted(env.evaluation_result_list, key=lambda x: (x[0], x[1])):
                    row_parts.append(f"{ds_name} {metric}: {value:.4f}")
                print("  ".join(row_parts))
            callbacks.append(_fmt_callback)
        else:
            callbacks.append(lgb.log_evaluation(period=-1))

        self.model_ = lgb.train(
            self._lgb_params(),
            dtrain,
            num_boost_round=self.num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=_rank_precision_lgb(),
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

    Each horizon (t+1, t+3, t+6) gets its own model because feature
    importance and optimal hyperparameters often differ across horizons.
    All models share the same backend (xgboost or lightgbm) and hyperparameters.

    Target column names match DataManager output:
        ``["ret_1m", "ret_3m", "ret_6m"]``

    Args:
        targets: List of target column names, one per horizon.
            Defaults to ``["ret_1m", "ret_3m", "ret_6m"]``.
        backend: ``"xgboost"`` or ``"lightgbm"``.
        **model_kwargs: Passed directly to XGBoostRanker or LGBMRanker.

    After fit(), access individual models via ``.models_[target]``.
    """

    def __init__(
        self,
        targets: list[str] | None = None,
        backend: str = "xgboost",
        **model_kwargs,
    ):
        if targets is None:
            targets = ["ret_1m", "ret_3m", "ret_6m"]
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
        Y: pd.DataFrame ,
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
            Y: Either a DataFrame with one column per target horizon
               (columns must match ``self.targets``, e.g. ``["ret_1m", "ret_3m",
               "ret_6m"]``), or a single Series / one-column DataFrame when
               training on a single horizon.
            groups: Stock counts per month, shared across all horizons.
                Built from ``df.groupby("yyyymm").size().tolist()``.
            eval_set: Optional ``(X_val, Y_val)`` for monitoring.
            eval_groups: Group sizes for the validation set.
        """
        missing = [t for t in self.targets if t not in Y.columns]
        if missing:
            raise ValueError(f"Target columns not found in Y: {missing}")

        feature_cols = [c for c in X.columns if c not in _META]

        self.models_: Dict[str, BaseRankingModel] = {}
        for target in self.targets:
            model = self._make_model()
            es = (eval_set[0][feature_cols], eval_set[1][target]) if eval_set is not None else None
            model.fit(
                X[feature_cols],
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
        feature_cols = [c for c in X.columns if c not in _META]
        return {target: model.predict(X[feature_cols]) for target, model in self.models_.items()}

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

    def get_history(self) -> Dict[str, dict]:
        """Returns training history (eval metrics per iteration) for each horizon."""
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        return {target: model.training_history_ for target, model in self.models_.items()}

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
