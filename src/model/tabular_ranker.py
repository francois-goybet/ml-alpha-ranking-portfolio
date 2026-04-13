"""Cross-sectional Tabular Ranker with row/column attention.

Architecture
------------
Inspired by TabICL (in-context learning for tabular data).

Tokenization
  Each feature value is projected into a D-dim vector via a learned per-feature
  linear map:
    x_tok[n, f] = W_val[f] · x[n, f] + b_val[f]   →  (N, F, D)

  W_val and b_val are (F, D) parameter tensors — one embedding per feature.

n_layers × CrossSectionalBlock
  Column attention  (cross-stock, per feature):
    For each feature f, all N stocks attend to each other.
    Learns: "is this stock's value for feature f extreme relative to this
    month's universe?"  — a learned, dynamic cross-sectional normalisation.

  Row attention  (cross-feature, per stock):
    For each stock n, all F features attend to each other.
    Learns: feature interactions per stock.

  Both use MultiheadAttention + residual + LayerNorm + FFN (pre-norm).

Pool over features
  mean(x_tok, dim=1)  →  (N, D)

Three parallel branches (same as MultiScaleMLPRanker)
  Each branch receives the pooled attended representation (N, D) and
  specialises for one return horizon via its BCE supervision:
    h1 → Linear(D,H) + LN + ReLU  →  aux_1  (BCE top-10% t+1, supervised by ret_1m)
    h2 → Linear(D,H) + LN + ReLU  →  aux_2  (BCE top-10% t+2, supervised by ret_3m)
    h3 → Linear(D,H) + LN + ReLU  →  aux_3  (BCE top-10% t+3, supervised by ret_6m)
    concat(h1,h2,h3)  →  Head  →  score

Group-aware inference
  Column attention MUST see the correct cross-sectional universe — all stocks
  in the same month group and no others.  Both fit() and predict() process
  each group individually.

  predict(X, groups) loops over groups.  If groups is None the entire X is
  treated as one group (useful for tiny test sets / smoke tests).

Losses
------
  α·BCE(aux1, top10_t1) + β·BCE(aux2, top10_t2) + γ·BCE(aux3, top10_t3)
  + δ·RankingLoss(score, rank_labels)
  + margin_weight·MarginLoss(score, rank_labels)

  ranking_loss options: "ndcg" | "lambda" | "softmax"
"""

import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.model.model import BaseRankingModel, _LABEL_ENCODERS
from src.model.gru_ranker import (
    _lambda_loss_group,
    _listnet_loss_group,
    _neural_ndcg_loss_group,
)
from src.model.mlp_ranker import _margin_loss_group


# ---------------------------------------------------------------------------
# Cross-sectional attention block
# ---------------------------------------------------------------------------

def _build_cs_block(d_model: int, n_heads: int, dropout: float):
    import torch.nn as nn

    class CrossSectionalBlock(nn.Module):
        """One layer: column attention → row attention → FFN, all with residuals."""

        def __init__(self):
            super().__init__()
            # Column: stocks attend to stocks (applied per-feature)
            self.col_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True)
            self.col_norm = nn.LayerNorm(d_model)
            # Row: features attend to features (applied per-stock)
            self.row_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True)
            self.row_norm = nn.LayerNorm(d_model)
            # FFN
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
            )

        def forward(self, x):
            # x: (N, F, D)
            # --- Column attention: reshape to (F, N, D) ---
            xc = x.permute(1, 0, 2)          # (F, N, D)  F batches of N tokens
            xc_out, _ = self.col_attn(xc, xc, xc)
            x = x + xc_out.permute(1, 0, 2)  # residual back to (N, F, D)
            x = self.col_norm(x)
            # --- Row attention: (N, F, D)  N batches of F tokens ---
            xr_out, _ = self.row_attn(x, x, x)
            x = x + xr_out                    # residual
            x = self.row_norm(x)
            # --- FFN ---
            x = x + self.ffn(x)
            return x                           # (N, F, D)

    return CrossSectionalBlock()


# ---------------------------------------------------------------------------
# Full network builder (called inside fit() once input_dim is known)
# ---------------------------------------------------------------------------

def _build_net(
    input_dim: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    hidden_dim: int,
    dropout: float,
):
    import torch
    import torch.nn as nn

    def _branch(in_d: int, out_d: int) -> nn.Sequential:
        layers = [nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    # Per-feature value embedding: (F, D) weight + bias
    val_weight = nn.Parameter(torch.empty(input_dim, d_model))
    val_bias   = nn.Parameter(torch.zeros(input_dim, d_model))
    nn.init.normal_(val_weight, std=0.02)

    cs_blocks = nn.ModuleList([_build_cs_block(d_model, n_heads, dropout)
                                for _ in range(n_layers)])

    # Three parallel branches (input: d_model after pooling)
    branch1 = _branch(d_model, hidden_dim)
    branch2 = _branch(d_model, hidden_dim)
    branch3 = _branch(d_model, hidden_dim)

    head = nn.Sequential(
        nn.Linear(3 * hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

    aux1 = nn.Linear(hidden_dim, 1)
    aux2 = nn.Linear(hidden_dim, 1)
    aux3 = nn.Linear(hidden_dim, 1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.val_weight = val_weight
            self.val_bias   = val_bias
            self.cs_blocks  = cs_blocks
            self.branch1    = branch1
            self.branch2    = branch2
            self.branch3    = branch3
            self.head       = head
            self.aux1       = aux1
            self.aux2       = aux2
            self.aux3       = aux3

        def forward(self, x):
            # x: (N, F)
            # --- Tokenization: (N, F) → (N, F, D) ---
            # x_tok[n, f] = val_weight[f] * x[n, f] + val_bias[f]
            tok = x.unsqueeze(-1) * self.val_weight.unsqueeze(0) + self.val_bias.unsqueeze(0)
            # --- Stacked cross-sectional blocks ---
            for blk in self.cs_blocks:
                tok = blk(tok)               # (N, F, D)
            # --- Pool over features → (N, D) ---
            pooled = tok.mean(dim=1)         # (N, D)
            # --- Parallel horizon branches ---
            h1 = self.branch1(pooled)        # (N, H) — t+1
            h2 = self.branch2(pooled)        # (N, H) — t+2
            h3 = self.branch3(pooled)        # (N, H) — t+3
            import torch as _t
            cat   = _t.cat([h1, h2, h3], dim=-1)      # (N, 3H)
            score = self.head(cat).squeeze(-1)          # (N,)
            a1    = self.aux1(h1).squeeze(-1)           # (N,)  BCE t+1
            a2    = self.aux2(h2).squeeze(-1)           # (N,)  BCE t+2
            a3    = self.aux3(h3).squeeze(-1)           # (N,)  BCE t+3
            return score, a1, a2, a3

    return Net()


# ---------------------------------------------------------------------------
# TabularRanker — BaseRankingModel-compatible
# ---------------------------------------------------------------------------

class TabularRanker(BaseRankingModel):
    """Cross-sectional tabular ranker with row/column attention.

    Parameters
    ----------
    d_model : int
        Token embedding dimension D.  Must be divisible by n_heads.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of stacked CrossSectionalBlock layers.
    hidden_dim : int
        Width H of each parallel horizon branch.
    dropout : float
        Dropout rate inside attention and branch blocks.
    alpha, beta_aux, gamma : float
        BCE loss weight for branch-1 (ret_1m), branch-2 (ret_3m),
        branch-3 (ret_6m) respectively.
    delta : float
        Ranking loss weight.
    margin_weight : float
        Weight of the STOSA-style margin regularizer (0 = disabled).
    margin : float
        Minimum score gap enforced between top-decile and rest stocks.
    ranking_loss : str
        "ndcg" | "lambda" | "softmax"
    ndcg_tau : float
        Temperature for NeuralSort (ndcg) or ListNet (softmax).
    lambda_sigma : float
        σ for LambdaLoss (ranking_loss="lambda" only).
    lr, weight_decay : float
        Adam optimiser hyper-parameters.
    num_rounds : int
        Number of training epochs.
    batch_groups : int
        Groups accumulated per gradient step (individual group forward
        passes are always used to keep column attention cross-sectionally
        correct).
    device : str
        "cpu" | "cuda" | "mps"
    label_encoder : str
        Relevance label encoder (same options as XGBoost backend).
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        alpha: float = 1.0,
        beta_aux: float = 0.7,
        gamma: float = 0.4,
        delta: float = 1.0,
        margin_weight: float = 0.0,
        margin: float = 1.0,
        ranking_loss: str = "ndcg",
        ndcg_tau: float = 1.0,
        lambda_sigma: float = 1.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 0.0,
        num_rounds: int = 50,
        batch_groups: int = 16,
        device: str = "cpu",
        label_encoder: str = "long_ranker",
        # BaseRankingModel compat kwargs (unused but accepted)
        learning_rate: float = 1e-3,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbosity: int = 0,
        eval_at: Optional[List[int]] = None,
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
        self.d_model       = d_model
        self.n_heads       = n_heads
        self.n_layers      = n_layers
        self.hidden_dim    = hidden_dim
        self.dropout       = dropout
        self.alpha         = alpha
        self.beta_aux      = beta_aux
        self.gamma         = gamma
        self.delta         = delta
        self.margin_weight = margin_weight
        self.margin        = margin
        self.ranking_loss  = ranking_loss
        self.ndcg_tau      = ndcg_tau
        self.lambda_sigma  = lambda_sigma
        self.lr            = float(lr)
        self.weight_decay  = float(weight_decay)
        self.max_grad_norm = max_grad_norm
        self.batch_groups  = batch_groups
        self.device        = device
        self.label_encoder = label_encoder

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "net_")

    def _get_device(self):
        import torch
        if self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    def _build_binary_targets(
        self,
        y: pd.DataFrame,
        groups: List[int],
        col: int,
        top_frac: float = 0.1,
    ) -> np.ndarray:
        """Binary top-decile labels for one horizon column."""
        series = y.iloc[:, col].values.astype(np.float32)
        out = np.zeros(len(series), dtype=np.float32)
        cursor = 0
        for g in groups:
            sl = slice(cursor, cursor + g)
            vals = series[sl]
            k = max(1, int(np.ceil(top_frac * g)))
            threshold = np.partition(vals, -k)[-k]
            out[sl] = (vals >= threshold).astype(np.float32)
            cursor += g
        return out

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        groups: List[int] | np.ndarray,
        eval_set=None,
        eval_groups=None,
        verbose: bool = False,
    ) -> "TabularRanker":
        import torch
        import torch.nn as nn

        if self.ranking_loss not in ("lambda", "ndcg", "softmax"):
            raise ValueError(
                f"ranking_loss must be 'lambda', 'ndcg', or 'softmax', "
                f"got '{self.ranking_loss}'"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        dev = self._get_device()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        X_arr = (X.values if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler()
        X_arr = self.scaler_.fit_transform(X_arr).astype(np.float32)

        groups = list(groups)
        encoder_fn  = _LABEL_ENCODERS.get(self.label_encoder, _LABEL_ENCODERS["long_ranker"])

        if isinstance(y, pd.DataFrame):
            y_primary = y.iloc[:, 0]
        else:
            y_primary = y if isinstance(y, pd.Series) else pd.Series(y)
        rank_labels = encoder_fn(y_primary, groups).astype(np.float32)

        if isinstance(y, pd.DataFrame) and y.shape[1] >= 3:
            bin1 = self._build_binary_targets(y, groups, 0)
            bin2 = self._build_binary_targets(y, groups, 1)
            bin3 = self._build_binary_targets(y, groups, 2)
        else:
            y_df = pd.DataFrame({"_": y_primary.values})
            bin1 = self._build_binary_targets(y_df, groups, 0)
            bin2, bin3 = bin1.copy(), bin1.copy()

        X_t    = torch.tensor(X_arr,       device=dev)
        rank_t = torch.tensor(rank_labels, device=dev)
        bin1_t = torch.tensor(bin1,        device=dev)
        bin2_t = torch.tensor(bin2,        device=dev)
        bin3_t = torch.tensor(bin3,        device=dev)

        input_dim = X_arr.shape[1]
        self.net_ = _build_net(
            input_dim, self.d_model, self.n_heads, self.n_layers,
            self.hidden_dim, self.dropout,
        ).to(dev)

        optimizer = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce_fn    = nn.BCEWithLogitsLoss()

        group_starts = np.concatenate([[0], np.cumsum(groups[:-1])])
        num_groups   = len(groups)
        self.training_history_: Dict[str, list] = {
            "loss": [], "bce": [], "rank": [], "margin": []}
        log_every = max(1, self.num_rounds // min(10, self.num_rounds))

        for epoch in range(self.num_rounds):
            self.net_.train()
            perm = np.random.permutation(num_groups)
            e_loss = e_bce = e_rank = e_margin = 0.0
            n_batches = 0

            for batch_start in range(0, num_groups, self.batch_groups):
                batch_idxs = perm[batch_start: batch_start + self.batch_groups]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=dev)
                b_bce = b_rank = b_margin = 0.0

                for gi in batch_idxs:
                    g_start = int(group_starts[gi])
                    g_size  = groups[gi]
                    sl      = slice(g_start, g_start + g_size)

                    x_g    = X_t[sl]         # (N_g, F)
                    rank_g = rank_t[sl]
                    b1, b2, b3 = bin1_t[sl], bin2_t[sl], bin3_t[sl]

                    score, a1, a2, a3 = self.net_(x_g)

                    loss_bce = (
                        self.alpha    * bce_fn(a1, b1)
                        + self.beta_aux * bce_fn(a2, b2)
                        + self.gamma    * bce_fn(a3, b3)
                    )

                    if self.ranking_loss == "ndcg":
                        rank_loss = _neural_ndcg_loss_group(
                            score, rank_g, tau=self.ndcg_tau)
                    elif self.ranking_loss == "softmax":
                        rank_loss = _listnet_loss_group(
                            score, rank_g, tau=self.ndcg_tau)
                    else:
                        rank_loss = _lambda_loss_group(
                            score, rank_g, sigma=self.lambda_sigma)

                    margin_loss = (
                        _margin_loss_group(score, rank_g, margin=self.margin)
                        if self.margin_weight > 0
                        else torch.tensor(0.0, device=dev)
                    )

                    g_loss = (
                        loss_bce
                        + self.delta * rank_loss
                        + self.margin_weight * margin_loss
                    )
                    batch_loss = batch_loss + g_loss
                    b_bce    += loss_bce.item()
                    b_rank   += rank_loss.item()
                    b_margin += margin_loss.item()

                n_g = len(batch_idxs)
                (batch_loss / n_g).backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.net_.parameters(), self.max_grad_norm)
                optimizer.step()

                e_loss   += (batch_loss.item() / n_g)
                e_bce    += b_bce   / n_g
                e_rank   += b_rank  / n_g
                e_margin += b_margin / n_g
                n_batches += 1

            nb = max(1, n_batches)
            avg_loss, avg_bce   = e_loss / nb, e_bce / nb
            avg_rank, avg_margin = e_rank / nb, e_margin / nb

            self.training_history_["loss"].append(avg_loss)
            self.training_history_["bce"].append(avg_bce)
            self.training_history_["rank"].append(avg_rank)
            self.training_history_["margin"].append(avg_margin)

            if verbose and (epoch % log_every == 0 or epoch == self.num_rounds - 1):
                margin_str = f"  margin={avg_margin:.4f}" if self.margin_weight > 0 else ""
                print(
                    f"  [TabRanker] epoch {epoch + 1:>4}/{self.num_rounds}"
                    f"  total={avg_loss:.4f}"
                    f"  bce={avg_bce:.4f}"
                    f"  {self.ranking_loss}={avg_rank:.4f}"
                    + margin_str
                )

        self.feature_names_ = feature_names
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        groups: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Return ranking scores.

        Parameters
        ----------
        X      : features (all rows)
        groups : list of group sizes.  If None, treats all X as one group.
                 Should be provided at inference so column attention sees the
                 correct cross-sectional universe per month.
        """
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        import torch

        dev   = self._get_device()
        X_arr = (X.values if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        X_arr = self.scaler_.transform(X_arr).astype(np.float32)
        X_t   = torch.tensor(X_arr, device=dev)

        if groups is None:
            groups = [len(X_arr)]

        self.net_.eval()
        out = np.empty(len(X_arr), dtype=np.float32)
        cursor = 0
        with torch.no_grad():
            for g_size in groups:
                x_g = X_t[cursor: cursor + g_size]
                score, _, _, _ = self.net_(x_g)
                out[cursor: cursor + g_size] = score.cpu().numpy()
                cursor += g_size

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    def get_feature_importance(self, importance_type: str = "gain") -> dict:
        return {}

    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        import torch
        torch.save({
            "net_state": self.net_.state_dict(),
            "scaler":    self.scaler_,
            "params":    {k: v for k, v in self.__dict__.items()
                          if not k.startswith("net_") and k != "scaler_"},
        }, path)

    @classmethod
    def load_model(cls, path: str) -> "TabularRanker":
        import torch
        ckpt    = torch.load(path, map_location="cpu")
        params  = ckpt["params"]
        init_keys = {
            "d_model", "n_heads", "n_layers", "hidden_dim", "dropout",
            "alpha", "beta_aux", "gamma", "delta", "margin_weight", "margin",
            "ranking_loss", "ndcg_tau", "lambda_sigma",
            "lr", "weight_decay", "max_grad_norm",
            "num_rounds", "batch_groups", "device", "label_encoder",
            "random_state",
        }
        m = cls(**{k: v for k, v in params.items() if k in init_keys})
        # Rebuild net with correct input_dim from saved weights
        val_w = ckpt["net_state"]["val_weight"]
        input_dim = val_w.shape[0]
        m.net_ = _build_net(
            input_dim, m.d_model, m.n_heads, m.n_layers, m.hidden_dim, m.dropout)
        m.net_.load_state_dict(ckpt["net_state"])
        m.scaler_ = ckpt["scaler"]
        return m
