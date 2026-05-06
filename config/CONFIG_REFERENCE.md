# Config Reference

All configuration lives in a single YAML file passed via `--config` (default: `config/config_francois.yaml`).
The file has six top-level sections.

---

## `data`

Controls data loading and train/val/test splitting.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `train_start` | date string | `"1990-01-01"` | First month included in the training set. |
| `train_end` | date string | `"2015-12-31"` | Last month included in the training set. |
| `val_start` | date string | `"2016-01-01"` | First month of the validation set (used for early stopping monitoring). |
| `val_end` | date string | `"2018-12-31"` | Last month of the validation set. |
| `test_start` | date string | `"2019-01-01"` | First month of the held-out test set (all portfolio metrics are computed here). |
| `test_end` | date string | `"2024-12-31"` | Last month of the test set. |
| `market_cap` | float | `10` | Minimum market cap filter in millions USD. Stocks below this threshold are dropped. |
| `top_market_cap` | int \| null | `null` | If set, keep only the top-N stocks by market cap per month. `null` keeps all. |
| `return_lags` | list[int] | `[1,2,3,6]` | Lagged monthly return columns to add as predictors (e.g. `[1,3,6]` adds `ret_lag1`, `ret_lag3`, `ret_lag6`). |
| `wrds_username` | string | `""` | WRDS username for raw data download. Only needed when rebuilding the dataset. |

---

## `feature_pipeline`

Each transform is applied in order: cross-sectional rank → winsorize → drop low-variance → impute → scale → ridge → PCA.

### Transforms

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cross_sectional_rank` | bool | `false` | Rank-normalise each feature cross-sectionally within each month to `[0, 1]`. Applied before winsorization. |
| `winsorize` | float \| null | `null` | Clip each feature at this percentile on each tail per month. E.g. `0.01` clips at 1st/99th percentile. `null` disables. |
| `drop_low_variance` | float \| null | `null` | Drop features whose variance is ≤ this threshold on the training set. `0.0` drops only constant features. `null` disables. |
| `impute` | `"median"` \| `"zero"` \| `"skip"` \| null | `"median"` | Fill NaN values. `"median"` uses the per-feature training median; `"zero"` fills with 0; `"skip"` leaves NaNs in place. |
| `scale` | `"standard"` \| `"minmax"` \| null | `null` | Feature scaling. `"standard"` → zero mean / unit variance per sector; `"minmax"` → [0,1] per sector. `null` disables. |
| `centroid_feature` | bool | `false` | Add a single feature equal to each stock's L2 distance from the monthly cross-sectional centroid. |
| `pca` | int \| null | `null` | Reduce features to this many principal components (fit on train, applied to val/test). `null` disables PCA. |

### `ridge_features` (optional sub-section)

Fits one Ridge regressor per target on training data using a **walk-forward** scheme (for month `t`, the Ridge is trained on months before `t` only — no look-ahead). Predictions are appended as new features `ridge_ret_1m`, `ridge_ret_3m`, `ridge_ret_6m`.

Remove the entire `ridge_features` key to disable.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `targets` | list[string] | — | Target columns to fit Ridge on. Usually `[ret_1m, ret_3m, ret_6m]`. |
| `alpha` | float | `1.0` | Ridge regularisation strength (higher = more regularised). |
| `min_train_months` | int | `24` | Minimum number of past months required before computing a ridge feature. Months with less history get NaN (imputed downstream). |

### `autoencoder` (optional sub-section)

Trains a variational or denoising autoencoder on the feature matrix and appends (or replaces) features with the latent codes.

Remove the entire `autoencoder` key to disable.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hidden_dims` | list[int] | `[32,16,8]` | Encoder layer sizes (decoder mirrors these in reverse). |
| `variational` | bool | `false` | Use a Variational Autoencoder (VAE). Mutually exclusive with `denoising`. |
| `denoising` | bool | `false` | Use a Denoising Autoencoder. Mutually exclusive with `variational`. |
| `beta` | float | `1.0` | β-VAE weight on the KL divergence term (only used when `variational: true`). |
| `epochs` | int | `50` | Training epochs. |
| `batch_size` | int | `512` | Mini-batch size. |
| `lr` | float | `1e-3` | Adam learning rate. |
| `weight_decay` | float | `1e-5` | L2 weight decay. |
| `dropout` | float | `0.1` | Dropout probability in encoder/decoder layers. |
| `replace_features` | bool | `false` | `true` → replace original features with latent codes only. `false` → append latent codes. |
| `device` | string | `"cpu"` | PyTorch device. `"mps"` for Apple Silicon, `"cuda"` for GPU, `"cpu"` otherwise. |

---

## `model`

Controls the ranking model. All keys except `backend` and `targets` are forwarded directly as constructor arguments to the chosen backend class.

### Backend selection

| Key | Type | Default | Options |
|-----|------|---------|---------|
| `backend` | string | `"xgboost"` | `"xgboost"`, `"lightgbm"`, `"ensemble"`, `"catboost"`, `"catboost_stack"` |
| `targets` | list[string] | `["ret_1m","ret_3m","ret_6m"]` | Target columns to train separate models for. One model per target. |

### Label encoding

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `label_encoder` | string \| null | `"argsort"` | How continuous returns are converted to relevance grades before training. See table below. |
| `downsample_top_k` | int \| null | `null` | When set (e.g. `30`), training rows are balanced per group: all top-k rows are kept, and an equal number of bottom rows are randomly sampled. Requires a label encoder that distinguishes top-k from rest (e.g. `"top30"`). Only supported by `catboost` and `catboost_stack` backends. |

**Label encoder options:**

| Value | Description |
|-------|-------------|
| `"argsort"` | Per-group integer ranks: 0 = worst, N−1 = best. |
| `"quintile"` | Per-group quintile labels 0–4 (4 = highest return). |
| `"decile"` | Per-group decile labels 0–9 (9 = highest return). |
| `"binary"` | 1 for stocks above the monthly median return, 0 otherwise. |
| `"long_ranker"` | Top-decile stocks get labels top_k..1; all others get 0. Focuses NDCG loss on the top 10%. |
| `"top30"` | Top-30 stocks per month get labels 30..1; all others get 0. Good with `downsample_top_k: 30`. |
| `null` | No encoding — pass raw continuous returns as labels. Supported natively by CatBoost YetiRank. |

### Shared hyperparameters (all backends)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_rounds` | int | `100` | Maximum number of boosting iterations. |
| `learning_rate` | float | `0.1` | Step size shrinkage (a.k.a. eta). Lower = slower but more robust. |
| `max_depth` | int | `6` | Maximum tree depth. |
| `subsample` | float | `0.8` | Row sampling fraction per tree. |
| `colsample_bytree` | float | `0.8` | Feature sampling fraction per tree. |
| `random_state` | int | `42` | Global random seed. |
| `verbosity` | int | `0` | Logging level. `0` = silent; higher = more output. |
| `eval_at` | list[int] | `[10,20]` | NDCG@k values to compute during training (XGBoost/LightGBM only). |

### XGBoost-specific (`backend: "xgboost"` or `"ensemble"`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `objective` | string | `"rank:pairwise"` | XGBoost objective. `"rank:pairwise"` or `"rank:ndcg"`. |
| `ndcg_exp_gain` | bool | `false` | Use exponential gain in NDCG computation. |

### LightGBM-specific (`backend: "lightgbm"`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `objective` | string | `"lambdarank"` | LightGBM objective. |
| `num_leaves` | int | `31` | Maximum number of leaves per tree. |
| `lambdarank_truncation_level` | int | `10` | Truncation level for LambdaRank. |
| `label_gain` | list[float] \| null | `null` | Maps integer relevance grades to NDCG gain values. `null` uses LightGBM's default `[0,1,3,7,15,...]`. |

### XGBoost ensemble-specific (`backend: "ensemble"`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_estimators` | int | `100` | Number of randomised XGBoost base models. |
| `feature_fraction` | [float, float] | `[0.3, 0.8]` | Min/max fraction of features each base model sees. |
| `depth_range` | [int, int] | `[3, 8]` | Min/max tree depth sampled per base model. |
| `lr_range` | [float, float] | `[0.01, 0.3]` | Min/max learning rate sampled log-uniformly per base model. |
| `time_fraction` | [float, float] | `[0.5, 1.0]` | Min/max fraction of training months each base model uses. |

### CatBoost-specific (`backend: "catboost"`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `early_stopping_rounds` | int | `10` | Stop training if eval metric does not improve for this many rounds. Requires `eval_set` (passed automatically from val data). |
| `l2_leaf_reg` | float | `3.0` | L2 regularisation on leaf values. |

### CatBoost stack ensemble-specific (`backend: "catboost_stack"`)

Includes all CatBoost-specific keys above, plus:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_estimators_l1` | int | `5` | Number of Layer-1 base models (each trained on a random feature subset). |
| `n_estimators_l2` | int | `3` | Number of Layer-2 meta-models (trained on stacked features: original + L1 OOF predictions). |
| `n_folds` | int | `5` | Temporal folds for walk-forward out-of-fold (OOF) generation. Fold 0 is always skipped (no past history). |
| `feature_fraction` | [float, float] | `[0.3, 0.8]` | Min/max fraction of features for L1 models. |
| `meta_feature_fraction` | [float, float] | `[0.5, 1.0]` | Min/max fraction of stacked features for L2 models. |
| `l1_exclude_cols` | list[string] \| null | `null` | Columns excluded from L1 feature selection (both OOF and full retrain). Use for globally-fit features like `ridge_ret_*` to prevent OOF look-ahead leakage. These columns remain available to L2 via the stacked matrix. |

**Total models trained** = `n_folds × n_estimators_l1` (OOF) `+ n_estimators_l1` (full retrain) `+ n_estimators_l2`.
With defaults: 5×5 + 5 + 3 = **33 CatBoost models** per target horizon.

---

## `ensemble`

Controls how predictions from the three target horizons (`ret_1m`, `ret_3m`, `ret_6m`) are combined into a single ranking score.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `combination` | `"mean_rank"` \| `"mean_score"` | `"mean_rank"` | `"mean_rank"` converts each horizon's scores to per-month ranks then averages (Borda count — robust to scale differences). `"mean_score"` averages raw scores directly. |
| `weights` | list[float] \| null | `null` | Per-horizon weights in the order `[ret_1m, ret_3m, ret_6m]`. `null` = equal weights. Example: `[1, 0, 0]` uses only the 1-month horizon. |

---

## `strategies`

A list of portfolio strategies to evaluate on the test set. Each strategy generates a monthly holdings DataFrame and is passed through `PortfolioAnalyzer` to compute PnL, Sharpe, drawdown, SP500 OLS regression, and Fama-French 5-factor regression.

**Available strategies:**

| Name | Description |
|------|-------------|
| `rf_only` | 100% risk-free asset (baseline). |
| `all_top_1` | Each month, hold the single highest-ranked stock with full weight. |
| `all_top_10_equal_weights` | Top-10 stocks by score, equally weighted. |
| `top_10_market_cap` | Top-10 stocks by score, weighted proportionally to market cap. |
| `top_20_market_cap` | Top-20 stocks by score, market-cap weighted. |
| `top_50_market_cap` | Top-50 stocks by score, market-cap weighted. |
| `top_100_market_cap` | Top-100 stocks by score, market-cap weighted. |
| `top_10_score_weighted` | Top-10 stocks, weighted proportionally to their (min-shifted) model score. |
| `top_20_score_weighted` | Top-20 stocks, score-weighted. |
| `top_50_score_weighted` | Top-50 stocks, score-weighted. |

Example:
```yaml
strategies: [rf_only, top_10_market_cap, top_10_score_weighted]
```

---

## `pipeline`

Miscellaneous runtime settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `experiment_name` | string \| null | `null` | Name shown in the Weights & Biases run. |
| `plots_dir` | string | `"generated/plots"` | Directory where plot files are saved locally. |
| `rebuild_dataset` | bool | `false` | If `true`, re-download raw data from WRDS even if a local cache exists. |
| `verbose` | bool | `true` | Print per-round boosting logs during training. |
| `bps` | float | `10` | Transaction cost in basis points applied to portfolio turnover. `10 bps = 0.10%` per trade. |

---

## Minimal working example

```yaml
data:
  train_start: 1990-01-01
  train_end:   2015-12-31
  val_start:   2016-01-01
  val_end:     2018-12-31
  test_start:  2019-01-01
  test_end:    2024-12-31
  market_cap:  10
  top_market_cap: 500

model:
  backend: catboost_stack
  targets: [ret_1m, ret_3m, ret_6m]
  n_estimators_l1: 5
  n_estimators_l2: 3
  n_folds: 5
  num_rounds: 50
  early_stopping_rounds: 10
  learning_rate: 0.1
  max_depth: 5
  subsample: 0.8
  colsample_bytree: 0.8
  l2_leaf_reg: 3.0
  label_encoder: top30
  downsample_top_k: 30
  feature_fraction: [0.3, 0.8]
  meta_feature_fraction: [0.5, 1.0]
  l1_exclude_cols: [ridge_ret_1m, ridge_ret_3m, ridge_ret_6m]

feature_pipeline:
  winsorize: 0.01
  impute: median
  scale: standard
  ridge_features:
    targets: [ret_1m, ret_3m, ret_6m]
    alpha: 1.0
    min_train_months: 24

ensemble:
  combination: mean_rank
  weights: null

strategies: [top_10_market_cap, top_10_score_weighted]

pipeline:
  experiment_name: my-run
  bps: 10
  verbose: false
```
