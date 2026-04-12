"""Feature transformation pipeline applied between data loading and model training.

Usage
-----
    pipeline = FeaturePipeline(config.get("feature_pipeline", {}))
    X_train = pipeline.fit_transform(X_train, groups=group_train, y=y_train)
    X_val   = pipeline.transform(X_val)
    X_test  = pipeline.transform(X_test)

Config keys (all optional)
--------------------------
feature_pipeline:
  cross_sectional_rank: true     # rank-normalise each feature within each month to [0, 1]
  winsorize: 0.01                # clip each feature at [q, 1-q] quantile per month (e.g. 0.01 = 1%)
  impute: median                 # fill NaNs: "median" | "zero" | null (no imputation)
  scale: standard                # global scaling after CS transforms: "standard" | "minmax" | null
  drop_low_variance: 0.0         # drop features whose variance < threshold (0.0 = drop constants only)
  pca: null                      # int | null — keep top-N PCA components (null = no PCA)
  centroid_feature: false        # bool — add L2 distance to cross-sectional centroid per month
  ridge_features:                # fit Ridge regressors on train labels and add predictions as features
    targets: [ret_1m, ret_3m, ret_6m]
    alpha: 1.0
  autoencoder:                   # train a feedforward autoencoder; append (or replace) features with latent codes
    latent_dim: 32
    hidden_dim: 128
    epochs: 50
    batch_size: 512
    lr: 1e-3
    weight_decay: 1e-5
    dropout: 0.1
    replace_features: false      # true = return only latent cols; false = append them
    device: null                 # null = cpu; "mps" for Apple Silicon
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.autoencoder import DenoisingTabularAutoencoder, TabularAutoencoder, VariationalTabularAutoencoder


class FeaturePipeline:
    """Stateful feature transformation pipeline.

    Call ``fit_transform`` on train, then ``transform`` on val/test.
    All cross-sectional steps (rank, winsorise) use the groups structure
    to operate within each month independently.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self._fitted = False

        # State fitted on train
        self._impute_values: pd.Series | None = None
        self._scale_mean: pd.Series | None = None
        self._scale_std: pd.Series | None = None
        self._scale_min: pd.Series | None = None
        self._scale_max: pd.Series | None = None
        self._kept_cols: list[str] | None = None
        self._pca = None
        self._ridge_models: dict | None = None       # {target: fitted Ridge}
        self._autoencoder: TabularAutoencoder | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def total_preprocessing_steps(
            self,
            X_train: pd.DataFrame,
            group_train: list[int],
            y_train: pd.DataFrame,
            X_val: pd.DataFrame,
            X_test: pd.DataFrame,    
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("\n--- Feature pipeline ---")
        print(f"  Steps: cs_rank={self.cfg.get('cross_sectional_rank', False)}  "
            f"winsorize={self.cfg.get('winsorize', None)}  "
            f"impute={self.cfg.get('impute', None)}  "
            f"scale={self.cfg.get('scale', None)}  "
            f"pca={self.cfg.get('pca', None)}  "
            f"centroid={self.cfg.get('centroid_feature', False)}  "
            f"ridge={list(self.cfg['ridge_features']['targets']) if self.cfg.get('ridge_features') else None}")
        X_train = self.fit_transform(X_train, groups=group_train, y=y_train)
        X_val   = self.transform(X_val)
        X_test  = self.transform(X_test)
        print(f"  Output shape: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
        return X_train, X_val, X_test
    def fit_transform(
        self,
        X: pd.DataFrame,
        groups: list[int],
        y: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fit on X (and optionally y for Ridge features) and return transformed copy."""
        _META = {"permno", "yyyymm", "ret", "ret_1m", "ret_3m", "ret_6m"}

        X = X.copy()

        meta = X[list(_META & set(X.columns))].copy()
        features = X.drop(columns=list(_META & set(X.columns)))

        # --- apply transformations ONLY on features ---
        features = self._apply_cs_transforms(features, groups, fit=True)
        features = self._add_centroid_feature(features, groups)
        features = self._drop_low_variance(features, fit=True)
        features = self._impute(features, fit=True)
        features = self._scale(features, fit=True)
        features = self._fit_ridge(features, y)
        # features = self._fit_autoencoder(features)
        features = self._apply_pca(features, fit=True)

        self._fitted = True

        # --- recombine ---
        return pd.concat([meta.reset_index(drop=True),
                        features.reset_index(drop=True)], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted transforms to val/test."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on the train set first.")

        _META = {"permno", "yyyymm", "ret", "ret_1m", "ret_3m", "ret_6m"}

        X = X.copy()

        meta = X[list(_META & set(X.columns))].copy()
        features = X.drop(columns=list(_META & set(X.columns)))

        # --- apply ONLY feature transforms ---
        features = self._apply_cs_transforms(features, groups=None, fit=False)
        features = self._add_centroid_feature(features, groups=None)
        features = self._select_cols(features)

        features = self._impute(features, fit=False)
        features = self._scale(features, fit=False)
        features = self._apply_ridge(features)
        # features = self._apply_autoencoder(features)
        features = self._apply_pca(features, fit=False)

        # --- recombine ---
        return pd.concat(
            [meta.reset_index(drop=True),
            features.reset_index(drop=True)],
            axis=1
        )

    # ------------------------------------------------------------------
    # Transform steps
    # ------------------------------------------------------------------

    def _apply_cs_transforms(
        self,
        X: pd.DataFrame,
        groups: list[int] | None,
        fit: bool,
    ) -> pd.DataFrame:
        """Cross-sectional winsorisation and rank-normalisation within each month."""
        winsorize_q = self.cfg.get("winsorize", None)
        do_rank = self.cfg.get("cross_sectional_rank", False)

        if not winsorize_q and not do_rank:
            return X

        if groups is None:
            # val/test: apply globally (no group info available at transform time)
            if winsorize_q:
                lo = X.quantile(winsorize_q)
                hi = X.quantile(1 - winsorize_q)
                X = X.clip(lower=lo, upper=hi, axis=1)
            if do_rank:
                X = X.rank(pct=True)
            return X

        # Train: apply within each cross-section (month)
        out_blocks = []
        cursor = 0
        for g in groups:
            sl = slice(cursor, cursor + g)
            block = X.iloc[sl].copy()
            if winsorize_q:
                lo = block.quantile(winsorize_q)
                hi = block.quantile(1 - winsorize_q)
                block = block.clip(lower=lo, upper=hi, axis=1)
            if do_rank:
                block = block.rank(pct=True)
            out_blocks.append(block)
            cursor += g
        return pd.concat(out_blocks, axis=0)

    def _drop_low_variance(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        threshold = self.cfg.get("drop_low_variance", None)
        if threshold is None:
            return X
        if fit:
            variances = X.var(skipna=True)
            self._kept_cols = variances[variances > threshold].index.tolist()
            dropped = len(X.columns) - len(self._kept_cols)
            if dropped:
                print(f"  [FeaturePipeline] Dropped {dropped} low-variance features (<= {threshold}).")
        return X[self._kept_cols] if self._kept_cols is not None else X

    def _select_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply column selection fitted during drop_low_variance."""
        if self._kept_cols is not None:
            missing = [c for c in self._kept_cols if c not in X.columns]
            if missing:
                raise ValueError(f"Val/test is missing columns: {missing}")
            return X[self._kept_cols]
        return X

    def _impute(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        strategy = self.cfg.get("impute", "median")
        if strategy is None:
            return X
        if strategy == "zero":
            return X.fillna(0.0)
        if strategy == "median":
            if fit:
                self._impute_values = X.median(skipna=True)
            return X.fillna(self._impute_values)
        raise ValueError(f"Unknown impute strategy '{strategy}'. Use 'median', 'zero', or null.")

    def _scale(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        strategy = self.cfg.get("scale", None)
        if strategy is None:
            return X
        if strategy == "standard":
            if fit:
                self._scale_mean = X.mean(skipna=True)
                self._scale_std = X.std(skipna=True).replace(0, 1)
            return (X - self._scale_mean) / self._scale_std
        if strategy == "minmax":
            if fit:
                self._scale_min = X.min(skipna=True)
                self._scale_max = X.max(skipna=True)
                denom = (self._scale_max - self._scale_min).replace(0, 1)
                self._scale_range = denom
            return (X - self._scale_min) / self._scale_range
        raise ValueError(f"Unknown scale strategy '{strategy}'. Use 'standard', 'minmax', or null.")

    def _add_centroid_feature(
        self,
        X: pd.DataFrame,
        groups: list[int] | None,
    ) -> pd.DataFrame:
        """Add L2 distance of each stock to the cross-sectional centroid per month.

        On val/test (groups=None) the distance is computed globally since we
        don't have month boundaries available at transform time.
        """
        if not self.cfg.get("centroid_feature", False):
            return X
        X_num = X.fillna(0).values.astype(float)
        dist = np.empty(len(X), dtype=float)
        if groups is not None:
            cursor = 0
            for g in groups:
                sl = slice(cursor, cursor + g)
                block = X_num[sl]
                centroid = block.mean(axis=0)
                dist[sl] = np.linalg.norm(block - centroid, axis=1)
                cursor += g
        else:
            centroid = X_num.mean(axis=0)
            dist = np.linalg.norm(X_num - centroid, axis=1)
        X = X.copy()
        X["centroid_dist"] = dist
        return X

    def _fit_ridge(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Fit one Ridge regressor per target and add predictions as features."""
        ridge_cfg = self.cfg.get("ridge_features", None)
        if ridge_cfg is None or y is None:
            return X
        from sklearn.linear_model import Ridge
        targets = ridge_cfg.get("targets", [])
        alpha = ridge_cfg.get("alpha", 1.0)
        self._ridge_models = {}
        X_arr = X.fillna(0).values
        for target in targets:
            if target not in y.columns:
                continue
            y_t = y[target].fillna(0).values
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_arr, y_t)
            self._ridge_models[target] = ridge
            X = X.copy()
            X[f"ridge_{target}"] = ridge.predict(X_arr)
        if self._ridge_models:
            print(f"  [FeaturePipeline] Ridge features added: {list(self._ridge_models.keys())}")
        return X

    def _apply_ridge(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted Ridge models to val/test."""
        if not self._ridge_models:
            return X
        X_arr = X.fillna(0).values
        for target, ridge in self._ridge_models.items():
            X = X.copy()
            X[f"ridge_{target}"] = ridge.predict(X_arr)
        return X

    def _fit_autoencoder(self, X: pd.DataFrame) -> pd.DataFrame:
        ae_cfg = self.cfg.get("autoencoder", None)
        if ae_cfg is None:
            return X
        if ae_cfg.get("variational", False):
            cls = VariationalTabularAutoencoder
        elif ae_cfg.get("denoising", False):
            cls = DenoisingTabularAutoencoder
        else:
            cls = TabularAutoencoder
        self._autoencoder = cls(ae_cfg)
        return self._autoencoder.fit_transform(X)

    def _apply_autoencoder(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._autoencoder is None:
            return X
        return self._autoencoder.transform(X)

    def _apply_pca(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        n_components = self.cfg.get("pca", None)
        if n_components is None:
            return X
        from sklearn.decomposition import PCA
        X_arr = X.fillna(0).values
        if fit:
            self._pca = PCA(n_components=n_components, random_state=42)
            X_arr = self._pca.fit_transform(X_arr)
            explained = self._pca.explained_variance_ratio_.sum()
            print(f"  [FeaturePipeline] PCA: kept {n_components} components, "
                  f"explained variance: {explained:.2%}")
        else:
            X_arr = self._pca.transform(X_arr)
        cols = [f"pc{i+1}" for i in range(n_components)]
        return pd.DataFrame(X_arr, index=X.index, columns=cols)
