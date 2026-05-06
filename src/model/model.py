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

try:
    from catboost import CatBoost as _CatBoost, Pool as _CatPool
except ImportError:
    _CatBoost = None
    _CatPool = None
    warnings.warn("CatBoost is not installed. Install it to use CatBoostRanker.")
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

def _recompute_groups(yyyymm_arr) -> list[int]:
    """Return stock counts per month from a (masked) yyyymm array.

    Equivalent to df.groupby("yyyymm").size().tolist() but works on a plain
    numpy array. Month order is preserved because yyyymm values are
    lexicographically sortable date strings (e.g. "199001").
    """
    _, counts = np.unique(yyyymm_arr, return_counts=True)
    return counts.tolist()

_META = {"permno", "yyyymm", "ret", "market_cap_musd", "sector", "ret_1m", "ret_3m", "ret_6m"}

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

class XGBoostEnsemble(BaseRankingModel):
    """Ensemble of ``n_estimators`` XGBoost rankers.

    Each base model is trained on a randomly drawn combination of:

    * **Feature subset** — a random fraction of columns sampled uniformly
      from ``feature_fraction`` range.
    * **Time window** — a random contiguous sub-interval of training months
      (requires ``yyyymm`` to be passed to ``fit``).
    * **max_depth** — integer sampled uniformly from ``depth_range``.
    * **learning_rate** — sampled log-uniformly from ``lr_range``.

    Final prediction is the **mean** of all base-ranker scores.

    Parameters
    ----------
    n_estimators : int
        Number of base XGBoost models to train. Default 100.
    feature_fraction : tuple[float, float]
        (min, max) fraction of features each model sees. Default (0.3, 0.8).
    depth_range : tuple[int, int]
        (min, max) inclusive range for ``max_depth``. Default (3, 8).
    lr_range : tuple[float, float]
        (min, max) for ``learning_rate`` on a log scale. Default (0.01, 0.3).
    time_fraction : tuple[float, float]
        (min, max) fraction of training months each model uses. Default (0.5, 1.0).
    All remaining kwargs are forwarded to XGBoostRanker as fixed hyperparameters.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        num_rounds: int = 10,
        objective: str = "rank:ndcg",
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        ndcg_exp_gain: bool = False,
        label_encoder: str | callable | None = "argsort",
        verbosity: int = 0,
        feature_fraction: tuple[float, float] = (0.3, 0.8),
        depth_range: tuple[int, int] = (3, 8),
        lr_range: tuple[float, float] = (0.01, 0.3),
        time_fraction: tuple[float, float] = (0.5, 1.0),
        eval_at: list[int] | None = None,
    ) -> None:
        super().__init__(
            num_rounds=num_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=verbosity,
        )
        self.n_estimators = n_estimators
        self.objective = objective
        self.ndcg_exp_gain = ndcg_exp_gain
        self.label_encoder = label_encoder
        self.feature_fraction = feature_fraction
        self.depth_range = depth_range
        self.lr_range = lr_range
        self.time_fraction = time_fraction

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "estimators_")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        groups: list[int] | np.ndarray,
        yyyymm: Optional[pd.Series | np.ndarray] = None,
        eval_set: Optional[tuple] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> "XGBoostEnsemble":
        """Train ``n_estimators`` XGBoostRankers with randomised configs.

        Args:
            X: Feature DataFrame (column names required).
            y: Target series (continuous returns).
            groups: Stock counts per month.
            yyyymm: Month identifier per row. Enables time-window sub-sampling.
            eval_set / eval_groups: Ignored (no early stopping in ensemble members).
            verbose: Print per-estimator summary if True.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame (column names required).")

        rng = np.random.RandomState(self.random_state)
        all_features = X.columns.tolist()
        n_features = len(all_features)
        y_s = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)

        yyyymm_arr = np.asarray(yyyymm) if yyyymm is not None else None
        all_months = np.unique(yyyymm_arr) if yyyymm_arr is not None else None

        self.estimators_: list[tuple[XGBoostRanker, list[str]]] = []

        for i in range(self.n_estimators):
            # --- Randomise hyperparameters ---
            depth = int(rng.randint(self.depth_range[0], self.depth_range[1] + 1))
            lr = float(np.exp(rng.uniform(
                np.log(self.lr_range[0]), np.log(self.lr_range[1])
            )))

            # --- Randomise feature subset ---
            frac = rng.uniform(self.feature_fraction[0], self.feature_fraction[1])
            n_sel = max(1, int(round(n_features * frac)))
            sel_idx = np.sort(rng.choice(n_features, size=n_sel, replace=False))
            selected_features = [all_features[j] for j in sel_idx]

            # --- Randomise time window ---
            if yyyymm_arr is not None and len(all_months) > 1:
                n_months = len(all_months)
                t_frac = rng.uniform(self.time_fraction[0], self.time_fraction[1])
                window = max(1, int(round(n_months * t_frac)))
                start = int(rng.randint(0, n_months - window + 1))
                sel_months = set(all_months[start: start + window])
                mask = np.isin(yyyymm_arr, list(sel_months))
                X_i = X.loc[mask, selected_features].reset_index(drop=True)
                y_i = y_s.loc[mask].reset_index(drop=True)
                groups_i = _recompute_groups(yyyymm_arr[mask])
                n_months_used = window
            else:
                X_i = X[selected_features]
                y_i = y_s
                groups_i = list(groups)
                n_months_used = len(all_months) if all_months is not None else "?"

            ranker = XGBoostRanker(
                num_rounds=self.num_rounds,
                objective=self.objective,
                learning_rate=lr,
                max_depth=depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=int(rng.randint(0, 2**31)),
                verbosity=self.verbosity,
                ndcg_exp_gain=self.ndcg_exp_gain,
                label_encoder=self.label_encoder,
            )
            ranker.fit(X_i, y_i, groups=groups_i, verbose=False)
            self.estimators_.append((ranker, selected_features))

            if verbose:
                print(
                    f"  [{i + 1:3d}/{self.n_estimators}] "
                    f"depth={depth}  lr={lr:.4f}  "
                    f"features={n_sel}/{n_features}  months={n_months_used}"
                )

        self.training_history_ = {}
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        scores = np.zeros(len(X), dtype=np.float64)
        for ranker, features in self.estimators_:
            scores += ranker.predict(X[features])
        return scores / len(self.estimators_)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        agg: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for ranker, _ in self.estimators_:
            for feat, val in ranker.get_feature_importance(importance_type).items():
                agg[feat] = agg.get(feat, 0.0) + val
                counts[feat] = counts.get(feat, 0) + 1
        return {f: agg[f] / counts[f] for f in agg}

    def save_model(self, filepath: str) -> None:
        raise NotImplementedError("Use pickle to save/load XGBoostEnsemble.")

    def load_model(self, filepath: str) -> "XGBoostEnsemble":
        raise NotImplementedError("Use pickle to save/load XGBoostEnsemble.")

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


def encode_labels_top30(y: pd.Series, groups: list[int], top_k: int = 30) -> np.ndarray:
    """Top-K focused labeling: top_k stocks get ranks top_k..1, all others get 0.

    Mirrors ``long_ranker`` but uses a fixed absolute count (default 30) rather
    than a fraction of the group, so the label distribution is consistent across
    months of different sizes.
    """
    out = np.zeros(len(y), dtype=np.int64)
    cursor = 0
    for g in groups:
        k = min(top_k, g)
        y_group = y.iloc[cursor : cursor + g].to_numpy()
        top_idx = np.argsort(y_group)[::-1][:k]
        for rank, idx in enumerate(top_idx):
            out[cursor + idx] = k - rank  # k, k-1, ..., 1
        cursor += g
    return out


def downsample_to_top_k(
    X: pd.DataFrame,
    y: pd.Series,
    groups: list[int],
    labels: np.ndarray,
    top_k: int = 30,
    rng: np.random.RandomState | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[int], np.ndarray]:
    """Balance each group by keeping all top-k rows and sampling top-k from the rest.

    For each group (month), retains the ``top_k`` stocks with label > 0 and
    randomly samples ``top_k`` stocks from the remainder (label == 0), so that
    each group contributes at most ``2 * top_k`` rows to training.  Groups
    smaller than ``2 * top_k`` are kept in full.

    Returns the filtered (X, y, groups, labels) tuple.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    keep_rows: list[np.ndarray] = []
    new_groups: list[int] = []
    cursor = 0
    for g in groups:
        sl = np.arange(cursor, cursor + g)
        top_mask = labels[sl] > 0
        top_idx = sl[top_mask]
        bot_idx = sl[~top_mask]
        n_sample = min(len(top_idx), len(bot_idx))
        if n_sample > 0 and len(bot_idx) > n_sample:
            sampled_bot = rng.choice(bot_idx, size=n_sample, replace=False)
        else:
            sampled_bot = bot_idx
        sel = np.sort(np.concatenate([top_idx, sampled_bot]))
        keep_rows.append(sel)
        new_groups.append(len(sel))
        cursor += g
    idx = np.concatenate(keep_rows)
    return (
        X.iloc[idx].reset_index(drop=True),
        y.iloc[idx].reset_index(drop=True),
        new_groups,
        labels[idx],
    )


def encode_labels_top_decile(y: pd.Series, groups: list[int]) -> np.ndarray:
    """Top-10% focused labeling: top decile gets descending ranks, rest gets 0.

    For each month: k = max(1, group_size // 10).
    Best stock → label k, second best → k-1, ..., k-th → 1, all others → 0.
    Identical to long_ranker but named explicitly for clarity.
    Works naturally with downsample_top_k: the balancing uses len(top stocks)
    which equals k = group_size // 10, so bottom is also sampled to ~10%.
    """
    return encode_labels_long_ranker(y, groups)


# Registry so callers can select an encoder by name from config
_LABEL_ENCODERS = {
    "quintile":     encode_labels_quintile,
    "decile":       encode_labels_decile,
    "binary":       encode_labels_binary,
    "argsort":      encode_labels_argsort,
    "long_ranker":  encode_labels_long_ranker,
    "top30":        encode_labels_top30,
    "top_decile":   encode_labels_top_decile,
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
# CatBoost YetiRank ranker
# ---------------------------------------------------------------------------

class CatBoostRanker(BaseRankingModel):
    """CatBoost YetiRank learning-to-rank model.

    YetiRank optimises a stochastic NDCG proxy using pairwise gradients.
    CatBoost accepts continuous relevance labels natively, so label_encoder
    defaults to None (raw forward returns are passed directly).

    Parameters
    ----------
    num_rounds : int
        Maximum boosting iterations. Default 50.
    early_stopping_rounds : int
        Stop if eval metric does not improve for this many rounds.
        Only active when eval_set is passed to fit(). Default 10.
    l2_leaf_reg : float
        L2 regularisation on leaf values. Default 3.0.
    label_encoder : str or callable or None
        How to convert continuous returns to relevance grades. Default None
        (pass continuous returns directly). Use 'argsort' for per-group ranks.
    """

    def __init__(
        self,
        num_rounds: int = 50,
        early_stopping_rounds: int = 10,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        verbosity: int = 0,
        eval_at: list[int] | None = None,
        label_encoder: str | callable | None = None,
        downsample_top_k: int | None = None,
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
        self.early_stopping_rounds = early_stopping_rounds
        self.l2_leaf_reg = l2_leaf_reg
        self.label_encoder = label_encoder
        self.downsample_top_k = downsample_top_k

    def _cb_params(self, with_early_stopping: bool = False) -> dict:
        p = {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=10",
            "iterations": self.num_rounds,
            "learning_rate": self.learning_rate,
            "depth": self.max_depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "bootstrap_type": "Bernoulli",
            "subsample": self.subsample,
            "rsm": self.colsample_bytree,
            "random_seed": self.random_state,
            "verbose": 0,
            "thread_count": -1,
        }
        if with_early_stopping:
            p["od_type"] = "Iter"
            p["od_wait"] = self.early_stopping_rounds
        return p

    def _resolve_encoder(self):
        if self.label_encoder is None:
            return None
        if callable(self.label_encoder) and not isinstance(self.label_encoder, str):
            return self.label_encoder
        if self.label_encoder in _LABEL_ENCODERS:
            return _LABEL_ENCODERS[self.label_encoder]
        raise ValueError(
            f"Unknown label_encoder '{self.label_encoder}'. "
            f"Choose from {list(_LABEL_ENCODERS)} or pass a callable."
        )

    def _build_pool(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: list[int] | np.ndarray,
        encoder_fn=None,
    ) -> "_CatPool":
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        y_s = y if isinstance(y, pd.Series) else pd.Series(y)
        y_arr = (
            encoder_fn(y_s, list(groups)).astype(np.float32)
            if encoder_fn is not None
            else y_s.values.astype(np.float32)
        )
        group_id = np.repeat(np.arange(len(groups)), groups).astype(np.uint32)
        feat_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        return _CatPool(data=X_arr, label=y_arr, group_id=group_id, feature_names=feat_names)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: list[int] | np.ndarray,
        eval_set: Optional[tuple] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> "CatBoostRanker":
        if _CatBoost is None:
            raise ImportError("CatBoost is not installed.")

        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        encoder_fn = self._resolve_encoder()
        y_s = y if isinstance(y, pd.Series) else pd.Series(y)

        # Optional: encode then downsample so bottom rows match top-k count per group
        if self.downsample_top_k is not None and encoder_fn is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("downsample_top_k requires X to be a DataFrame.")
            labels = encoder_fn(y_s, list(groups))
            X, y_s, groups, labels = downsample_to_top_k(
                X, y_s, list(groups), labels, top_k=self.downsample_top_k,
                rng=np.random.RandomState(self.random_state),
            )
            dtrain = self._build_pool(X, y_s, groups, encoder_fn=None)
            dtrain_pool = _CatPool(
                data=X.values.astype(np.float32),
                label=labels.astype(np.float32),
                group_id=np.repeat(np.arange(len(groups)), groups).astype(np.uint32),
                feature_names=feature_names,
            )
            dtrain = dtrain_pool
        else:
            dtrain = self._build_pool(X, y_s, groups, encoder_fn)

        has_eval = eval_set is not None and eval_groups is not None
        params = self._cb_params(with_early_stopping=has_eval)
        if verbose:
            params["verbose"] = max(1, self.num_rounds // 10)

        fit_kwargs: dict = {"X": dtrain}
        if has_eval:
            X_e, y_e = eval_set
            fit_kwargs["eval_set"] = self._build_pool(X_e, y_e, eval_groups, encoder_fn)

        self.model_ = _CatBoost(params)
        self.model_.fit(**fit_kwargs)
        self.feature_names_ = feature_names
        self.training_history_ = {}
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        return self.model_.predict(X_arr)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        cb_type = "PredictionValuesChange" if importance_type == "gain" else "InternalFeatureImportance"
        names = self.model_.feature_names_
        scores = self.model_.get_feature_importance(type=cb_type)
        return dict(zip(names, scores.tolist()))

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        self.model_.save_model(filepath)

    def load_model(self, filepath: str) -> "CatBoostRanker":
        if _CatBoost is None:
            raise ImportError("CatBoost is not installed.")
        self.model_ = _CatBoost()
        self.model_.load_model(filepath)
        return self


# ---------------------------------------------------------------------------
# CatBoost 2-layer stacking ensemble (AutoGluon-style)
# ---------------------------------------------------------------------------

class CatBoostStackEnsemble(BaseRankingModel):
    """AutoGluon-style 2-layer stacking ensemble backed by CatBoost YetiRank.

    Architecture
    ------------
    Layer 1 — ``n_estimators_l1`` CatBoostRankers trained on random feature
    subsets of X.  Out-of-fold (OOF) predictions are built via ``n_folds``
    temporal cross-validation to avoid look-ahead leakage.

    Stacking — OOF predictions are appended to X, producing a stacked matrix
    of shape (n_rows, n_features + n_estimators_l1).

    Layer 2 — ``n_estimators_l2`` CatBoostRankers (the "dense" meta-learners)
    each trained on a random feature subset of the stacked matrix.

    Final prediction — mean of all Layer 2 scores.

    After fit(), Layer 1 models are retrained on the full training data so that
    at inference time they exploit all available history.

    Look-ahead note
    ---------------
    Features produced by a globally-fit transform (e.g. ``ridge_ret_*`` from
    ``FeaturePipeline``) carry inherent look-ahead within OOF folds because the
    Ridge was fitted on the entire training set.  Pass those column names via
    ``l1_exclude_cols`` to keep them out of Layer 1 feature pools entirely.
    They will still appear in the stacked matrix passed to Layer 2, where they
    are legitimate (L2 is trained on the same time-ordered data the Ridge was
    fitted on).

    Parameters
    ----------
    n_estimators_l1 : int
        Number of first-layer base models. Default 5.
    n_estimators_l2 : int
        Number of second-layer meta-models. Default 3.
    n_folds : int
        Temporal folds for OOF generation. Default 5.
    feature_fraction : tuple[float, float]
        (min, max) fraction of original features per L1 model. Default (0.3, 0.8).
    meta_feature_fraction : tuple[float, float]
        (min, max) fraction of stacked features per L2 model. Default (0.5, 1.0).
    num_rounds : int
        Max CatBoost iterations per model. Default 50.
    early_stopping_rounds : int
        Early stopping patience (requires eval_set in fit). Default 10.
    l1_exclude_cols : list[str] or None
        Column names to exclude from Layer 1 feature selection (both OOF and
        full retrain).  Use this for any columns whose values were derived from
        a global fit over the entire training period (e.g. ``["ridge_ret_1m",
        "ridge_ret_3m", "ridge_ret_6m"]``).  These columns are still included
        in the stacked feature matrix seen by Layer 2.  Default None.
    """

    def __init__(
        self,
        n_estimators_l1: int = 5,
        n_estimators_l2: int = 3,
        n_folds: int = 5,
        feature_fraction: tuple[float, float] = (0.3, 0.8),
        meta_feature_fraction: tuple[float, float] = (0.5, 1.0),
        num_rounds: int = 50,
        early_stopping_rounds: int = 10,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        verbosity: int = 0,
        eval_at: list[int] | None = None,
        label_encoder: str | callable | None = None,
        l1_exclude_cols: list[str] | None = None,
        downsample_top_k: int | None = None,
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
        self.n_estimators_l1 = n_estimators_l1
        self.n_estimators_l2 = n_estimators_l2
        self.n_folds = n_folds
        self.feature_fraction = feature_fraction
        self.meta_feature_fraction = meta_feature_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.l2_leaf_reg = l2_leaf_reg
        self.label_encoder = label_encoder
        self.l1_exclude_cols: set[str] = set(l1_exclude_cols) if l1_exclude_cols else set()
        self.downsample_top_k = downsample_top_k

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "l1_estimators_")

    def _make_ranker(self, rs: int) -> CatBoostRanker:
        return CatBoostRanker(
            num_rounds=self.num_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            l2_leaf_reg=self.l2_leaf_reg,
            random_state=rs,
            verbosity=self.verbosity,
            eval_at=self.eval_at,
            label_encoder=self.label_encoder,
            downsample_top_k=self.downsample_top_k,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        groups: list[int] | np.ndarray,
        yyyymm: Optional[pd.Series | np.ndarray] = None,
        eval_set: Optional[tuple] = None,
        eval_groups: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> "CatBoostStackEnsemble":
        if _CatBoost is None:
            raise ImportError("CatBoost is not installed.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame (column names required).")

        rng = np.random.RandomState(self.random_state)
        all_features = X.columns.tolist()
        y_s = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)
        yyyymm_arr = np.asarray(yyyymm) if yyyymm is not None else np.arange(len(X))

        # ── 1. Fixed feature subsets for L1 ────────────────────────────────
        # Exclude globally-fit features (e.g. ridge_*) from L1 to avoid OOF
        # leakage. These columns are still present in the stacked matrix for L2.
        l1_candidate_features = [f for f in all_features if f not in self.l1_exclude_cols]
        n_l1_candidates = len(l1_candidate_features)
        l1_feature_sets: list[list[str]] = []
        for _ in range(self.n_estimators_l1):
            frac = rng.uniform(self.feature_fraction[0], self.feature_fraction[1])
            n_sel = max(1, int(round(n_l1_candidates * frac)))
            idxs = np.sort(rng.choice(n_l1_candidates, size=n_sel, replace=False))
            l1_feature_sets.append([l1_candidate_features[j] for j in idxs])

        # ── 2. Walk-forward OOF split → build stacking features ────────────
        # For fold k: train on months 0..k-1 only (strict past), predict on k.
        # Fold 0 has no past history and is skipped; those rows get no OOF
        # prediction and are excluded from L2 training to avoid leakage.
        unique_months = np.unique(yyyymm_arr)
        n_months = len(unique_months)
        fold_size = max(1, n_months // self.n_folds)
        oof_preds = np.zeros((len(X), self.n_estimators_l1), dtype=np.float64)
        oof_valid = np.zeros(len(X), dtype=bool)  # rows that received valid OOF preds

        X_idx = np.arange(len(X))
        for fold_idx in range(self.n_folds):
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < self.n_folds - 1 else n_months
            past_months = unique_months[:start]   # strictly before validation window

            if len(past_months) == 0:
                continue  # no history yet — skip to avoid leakage

            val_months = unique_months[start:end]
            train_mask = np.isin(yyyymm_arr, past_months)
            val_mask = np.isin(yyyymm_arr, val_months)

            X_tr = X.iloc[train_mask].reset_index(drop=True)
            y_tr = y_s.iloc[train_mask].reset_index(drop=True)
            groups_tr = _recompute_groups(yyyymm_arr[train_mask])

            X_va = X.iloc[val_mask].reset_index(drop=True)
            val_positions = X_idx[val_mask]

            for est_idx, feats in enumerate(l1_feature_sets):
                ranker = self._make_ranker(int(rng.randint(0, 2**31)))
                ranker.fit(X_tr[feats], y_tr, groups=groups_tr, verbose=False)
                oof_preds[val_positions, est_idx] = ranker.predict(X_va[feats])

            oof_valid[val_positions] = True
            if verbose:
                print(f"  OOF fold {fold_idx + 1}/{self.n_folds} complete")

        # ── 3. Stacked feature matrix (only rows with valid OOF preds) ───────
        oof_col_names = [f"_l1_{i}" for i in range(self.n_estimators_l1)]
        stacked_X_full = pd.DataFrame(
            np.hstack([X.values, oof_preds]),
            columns=all_features + oof_col_names,
            index=X.index,
        )
        stacked_features = stacked_X_full.columns.tolist()
        n_stacked = len(stacked_features)

        # Restrict L2 training to rows where OOF was generated without leakage
        stacked_X_l2 = stacked_X_full.iloc[oof_valid].reset_index(drop=True)
        y_l2 = y_s.iloc[oof_valid].reset_index(drop=True)
        groups_l2 = _recompute_groups(yyyymm_arr[oof_valid])

        # ── 4. Train L2 meta-models on stacked features ────────────────────
        self.l2_estimators_: list[tuple[CatBoostRanker, list[str]]] = []
        for j in range(self.n_estimators_l2):
            frac = rng.uniform(self.meta_feature_fraction[0], self.meta_feature_fraction[1])
            n_sel = max(1, int(round(n_stacked * frac)))
            idxs = np.sort(rng.choice(n_stacked, size=n_sel, replace=False))
            sel_feats = [stacked_features[k] for k in idxs]
            ranker = self._make_ranker(int(rng.randint(0, 2**31)))
            ranker.fit(stacked_X_l2[sel_feats], y_l2, groups=groups_l2, verbose=False)
            self.l2_estimators_.append((ranker, sel_feats))
            if verbose:
                print(f"  L2 model {j + 1}/{self.n_estimators_l2} complete")

        # ── 5. Retrain L1 on full data ──────────────────────────────────────
        self.l1_estimators_: list[tuple[CatBoostRanker, list[str]]] = []
        for i, feats in enumerate(l1_feature_sets):
            ranker = self._make_ranker(int(rng.randint(0, 2**31)))
            ranker.fit(X[feats], y_s, groups=list(groups), verbose=False)
            self.l1_estimators_.append((ranker, feats))
            if verbose:
                print(f"  L1 model {i + 1}/{self.n_estimators_l1} retrained on full data")

        self.stacked_features_ = stacked_features
        self.training_history_ = {}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        l1_preds = np.stack(
            [ranker.predict(X[feats]) for ranker, feats in self.l1_estimators_], axis=1
        )
        oof_col_names = [f"_l1_{i}" for i in range(self.n_estimators_l1)]
        stacked_X = pd.DataFrame(
            np.hstack([X.values, l1_preds]),
            columns=X.columns.tolist() + oof_col_names,
            index=X.index,
        )

        l2_scores = np.zeros(len(X), dtype=np.float64)
        for ranker, feats in self.l2_estimators_:
            l2_scores += ranker.predict(stacked_X[feats])
        return l2_scores / len(self.l2_estimators_)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        agg: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        all_estimators = list(self.l1_estimators_) + list(self.l2_estimators_)
        for ranker, _ in all_estimators:
            for feat, val in ranker.get_feature_importance(importance_type).items():
                if feat.startswith("_l1_"):
                    continue
                agg[feat] = agg.get(feat, 0.0) + val
                counts[feat] = counts.get(feat, 0) + 1
        return {f: agg[f] / counts[f] for f in agg}

    def save_model(self, filepath: str) -> None:
        raise NotImplementedError("Use pickle to save/load CatBoostStackEnsemble.")

    def load_model(self, filepath: str) -> "CatBoostStackEnsemble":
        raise NotImplementedError("Use pickle to save/load CatBoostStackEnsemble.")


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
        if backend not in ("xgboost", "lightgbm", "ensemble", "catboost", "catboost_stack"):
            raise ValueError("backend must be 'xgboost', 'lightgbm', 'ensemble', 'catboost', or 'catboost_stack'.")
        self.targets = targets
        self.backend = backend
        self.model_kwargs = model_kwargs

    def _make_model(self) -> BaseRankingModel:
        if self.backend == "xgboost":
            return XGBoostRanker(**self.model_kwargs)
        if self.backend == "lightgbm":
            return LGBMRanker(**self.model_kwargs)
        if self.backend == "ensemble":
            return XGBoostEnsemble(**self.model_kwargs)
        if self.backend == "catboost":
            return CatBoostRanker(**self.model_kwargs)
        if self.backend == "catboost_stack":
            return CatBoostStackEnsemble(**self.model_kwargs)

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
            extra = {}
            if self.backend in ("ensemble", "catboost_stack") and "yyyymm" in X.columns:
                extra["yyyymm"] = X["yyyymm"]
            model.fit(
                X[feature_cols],
                Y[target],
                groups=groups,
                eval_set=es,
                eval_groups=eval_groups,
                verbose=verbose,
                **extra,
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
