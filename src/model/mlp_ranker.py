"""Multi-scale MLP ranking model.

Architecture
------------
Three parallel branches, each processing the raw input independently:

  Input (F)
    ├─ Block-1 : Linear(F,H) + LayerNorm + ReLU + Dropout → h1  → aux_1 (BCE: top-10% at t+1)
    ├─ Block-2 : Linear(F,H) + LayerNorm + ReLU + Dropout → h2  → aux_2 (BCE: top-10% at t+2)
    └─ Block-3 : Linear(F,H) + LayerNorm + ReLU + Dropout → h3  → aux_3 (BCE: top-10% at t+3)

  concat(h1, h2, h3)  →  Head: Linear(3H,H) + ReLU + Linear(H,1)  →  score

Each branch is supervised by the BCE loss for its own horizon, so it learns a
representation specifically useful for that prediction horizon. The final head then
combines all three horizon-aware representations into a single ranking score.

Losses
------
  α·BCE(aux_1, top10_t1) + β·BCE(aux_2, top10_t2) + γ·BCE(aux_3, top10_t3)
  + δ·RankingLoss(score, rank_labels)

  ranking_loss options:
    "ndcg"    — NeuralNDCG via NeuralSort  (default)
    "lambda"  — LambdaLoss  (pairwise, |ΔNDCG|-weighted)
    "softmax" — ListNet top-1 cross-entropy  (fastest, O(N))

At inference only the final ranking score is returned.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.model.model import BaseRankingModel, _LABEL_ENCODERS
from src.model.gru_ranker import (
    _lambda_loss_group,
    _listnet_loss_group,
    _neural_ndcg_loss_group,
)


# ---------------------------------------------------------------------------
# Margin regularizer (STOSA-style)
# ---------------------------------------------------------------------------

def _margin_loss_group(scores, labels, margin: float = 1.0):
    """Margin regularizer for a single ranking group.

    For every (positive, negative) pair in the group, penalises cases where
    the score gap is smaller than `margin`:

        L = mean( max(0, margin - (score_pos - score_neg)) )

    With `long_ranker` labels, positives = top-10% stocks (label > 0) and
    negatives = the rest (label == 0).  This directly enforces the separation
    that Hit@10 measures: the model must predict top-decile stocks with scores
    at least `margin` higher than non-top-decile stocks.

    Parameters
    ----------
    scores : torch.Tensor (N,)
    labels : torch.Tensor (N,)  — non-negative relevance grades
    margin : float

    Returns
    -------
    torch.Tensor — scalar loss (0 if no positive/negative items in group)
    """
    import torch

    pos_mask = labels > 0
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return scores.sum() * 0.0

    pos_scores = scores[pos_mask]   # (P,)
    neg_scores = scores[neg_mask]   # (Q,)

    # All P×Q gaps in one broadcast operation
    gaps = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # (P, Q)
    violations = torch.clamp(margin - gaps, min=0.0)          # (P, Q)
    return violations.mean()


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------

def _build_net(input_dim: int, hidden_dim: int, dropout: float):
    import torch.nn as nn

    def _block(in_d: int, out_d: int) -> nn.Sequential:
        layers: list = [nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    # Three parallel branches — each receives the raw input (F) directly.
    # block1 specialises for t+1, block2 for t+2, block3 for t+3.
    block1 = _block(input_dim, hidden_dim)   # → h1  supervised by BCE(t+1)
    block2 = _block(input_dim, hidden_dim)   # → h2  supervised by BCE(t+2)
    block3 = _block(input_dim, hidden_dim)   # → h3  supervised by BCE(t+3)

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
            self.block1 = block1
            self.block2 = block2
            self.block3 = block3
            self.head    = head
            self.aux1    = aux1
            self.aux2    = aux2
            self.aux3    = aux3

        def forward(self, x):
            h1    = self.block1(x)                          # (N, H) — t+1 branch
            h2    = self.block2(x)                          # (N, H) — t+2 branch
            h3    = self.block3(x)                          # (N, H) — t+3 branch
            import torch
            cat   = torch.cat([h1, h2, h3], dim=-1)        # (N, 3H)
            score = self.head(cat).squeeze(-1)              # (N,)
            a1    = self.aux1(h1).squeeze(-1)               # (N,)  BCE target: top-10% t+1
            a2    = self.aux2(h2).squeeze(-1)               # (N,)  BCE target: top-10% t+2
            a3    = self.aux3(h3).squeeze(-1)               # (N,)  BCE target: top-10% t+3
            return score, a1, a2, a3

    return Net()


# ---------------------------------------------------------------------------
# MultiScaleMLPRanker
# ---------------------------------------------------------------------------

class MultiScaleMLPRanker(BaseRankingModel):
    """Multi-scale MLP ranker with NeuralNDCG + auxiliary BCE heads.

    Parameters
    ----------
    hidden_dim : int
        Width of each MLP block (H).  Default 128.
    dropout : float
        Dropout rate after each block.  Default 0.1.
    alpha, beta_aux, gamma : float
        BCE loss weights for t+1, t+2, t+3 auxiliary heads.
    delta : float
        Weight on the ranking loss term.
    ranking_loss : str
        ``"ndcg"`` (default) | ``"lambda"`` | ``"softmax"``.
    ndcg_tau : float
        Temperature for NeuralSort (ndcg) or ListNet softmax (softmax).
    lambda_sigma : float
        σ parameter for LambdaLoss.  Ignored when ranking_loss != 'lambda'.
    lr : float
        Adam learning rate.
    weight_decay : float
        Adam L2 regularisation.
    num_rounds : int
        Training epochs.
    batch_groups : int
        Monthly groups per gradient step.
    device : str
        ``"cpu"`` | ``"cuda"`` | ``"mps"``.
    label_encoder : str
        Rank-label encoder for the ranking loss.  Default ``"long_ranker"``.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        alpha: float = 1.0,
        beta_aux: float = 0.7,
        gamma: float = 0.4,
        delta: float = 1.0,
        margin_weight: float = 0.0,  # weight of margin regularizer (0 = disabled)
        margin: float = 1.0,          # minimum score gap between positive and negative items
        ranking_loss: str = "ndcg",
        ndcg_tau: float = 1.0,
        lambda_sigma: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 0.0,
        num_rounds: int = 50,
        batch_groups: int = 8,
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

    @staticmethod
    def _build_binary_targets(y: pd.DataFrame, groups: List[int], col_idx: int) -> np.ndarray:
        out  = np.zeros(len(y), dtype=np.float32)
        vals = y.iloc[:, col_idx].values
        cursor = 0
        for g in groups:
            top_k = max(1, g // 10)
            top_idx = np.argsort(vals[cursor:cursor+g])[::-1][:top_k]
            out[cursor + top_idx] = 1.0
            cursor += g
        return out

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | pd.Series,
        groups: List[int] | np.ndarray,
        eval_set=None,
        eval_groups=None,
        verbose: bool = False,
    ) -> "MultiScaleMLPRanker":
        import torch
        import torch.nn as nn

        if self.ranking_loss not in ("lambda", "ndcg", "softmax"):
            raise ValueError(
                f"ranking_loss must be 'lambda', 'ndcg', or 'softmax', got '{self.ranking_loss}'"
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

        # Rank labels for the final head ranking loss — use ret_1m (primary target).
        encoder_fn = _LABEL_ENCODERS.get(self.label_encoder, _LABEL_ENCODERS["long_ranker"])

        if isinstance(y, pd.DataFrame):
            y_primary = y.iloc[:, 0]
        else:
            y_primary = y if isinstance(y, pd.Series) else pd.Series(y)
        rank_labels = encoder_fn(y_primary, groups).astype(np.float32)

        # Binary targets for auxiliary BCE heads
        if isinstance(y, pd.DataFrame) and y.shape[1] >= 3:
            bin1 = self._build_binary_targets(y, groups, 0)
            bin2 = self._build_binary_targets(y, groups, 1)
            bin3 = self._build_binary_targets(y, groups, 2)
        else:
            y_df = pd.DataFrame({"_": y_primary.values})
            bin1 = self._build_binary_targets(y_df, groups, 0)
            bin2, bin3 = bin1.copy(), bin1.copy()

        X_t      = torch.tensor(X_arr,      device=dev)
        rank_t   = torch.tensor(rank_labels, device=dev)
        bin1_t   = torch.tensor(bin1,        device=dev)
        bin2_t   = torch.tensor(bin2,        device=dev)
        bin3_t   = torch.tensor(bin3,        device=dev)

        input_dim  = X_arr.shape[1]
        self.net_  = _build_net(input_dim, self.hidden_dim, self.dropout).to(dev)
        optimizer  = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce_fn     = nn.BCEWithLogitsLoss()

        group_starts = np.concatenate([[0], np.cumsum(groups[:-1])])
        num_groups   = len(groups)
        self.training_history_: Dict[str, list] = {"loss": [], "bce": [], "rank": [], "margin": []}
        log_every = max(1, self.num_rounds // min(10, self.num_rounds))

        for epoch in range(self.num_rounds):
            self.net_.train()
            perm    = np.random.permutation(num_groups)
            e_loss = e_bce = e_rank = e_margin = 0.0
            n_batches = 0

            for batch_start in range(0, num_groups, self.batch_groups):
                batch_idxs = perm[batch_start: batch_start + self.batch_groups]
                row_indices = np.concatenate([
                    np.arange(group_starts[gi], group_starts[gi] + groups[gi])
                    for gi in batch_idxs
                ])
                row_t = torch.tensor(row_indices, dtype=torch.long, device=dev)

                x_b    = X_t[row_t]
                rank_b = rank_t[row_t]
                b1, b2, b3 = bin1_t[row_t], bin2_t[row_t], bin3_t[row_t]
                batch_gs   = [groups[gi] for gi in batch_idxs]

                score, a1, a2, a3 = self.net_(x_b)

                loss_bce = (
                    self.alpha    * bce_fn(a1, b1)
                    + self.beta_aux * bce_fn(a2, b2)
                    + self.gamma    * bce_fn(a3, b3)
                )

                rank_total   = torch.tensor(0.0, device=dev)
                margin_total = torch.tensor(0.0, device=dev)
                cursor = 0
                for g_size in batch_gs:
                    sl = slice(cursor, cursor + g_size)
                    if self.ranking_loss == "ndcg":
                        rank_total = rank_total + _neural_ndcg_loss_group(
                            score[sl], rank_b[sl], tau=self.ndcg_tau)
                    elif self.ranking_loss == "softmax":
                        rank_total = rank_total + _listnet_loss_group(
                            score[sl], rank_b[sl], tau=self.ndcg_tau)
                    else:
                        rank_total = rank_total + _lambda_loss_group(
                            score[sl], rank_b[sl], sigma=self.lambda_sigma)
                    if self.margin_weight > 0:
                        margin_total = margin_total + _margin_loss_group(
                            score[sl], rank_b[sl], margin=self.margin)
                    cursor += g_size
                rank_avg   = rank_total   / len(batch_gs)
                margin_avg = margin_total / len(batch_gs)

                total_loss = loss_bce + self.delta * rank_avg + self.margin_weight * margin_avg
                optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.net_.parameters(), self.max_grad_norm)
                optimizer.step()

                e_loss   += total_loss.item()
                e_bce    += loss_bce.item()
                e_rank   += rank_avg.item()
                e_margin += margin_avg.item()
                n_batches += 1

            nb = max(1, n_batches)
            avg_loss, avg_bce, avg_rank, avg_margin = e_loss/nb, e_bce/nb, e_rank/nb, e_margin/nb
            self.training_history_["loss"].append(avg_loss)
            self.training_history_["bce"].append(avg_bce)
            self.training_history_["rank"].append(avg_rank)
            self.training_history_["margin"].append(avg_margin)

            if verbose and (epoch % log_every == 0 or epoch == self.num_rounds - 1):
                margin_str = f"  margin={avg_margin:.4f}" if self.margin_weight > 0 else ""
                print(
                    f"  [MLPRanker] epoch {epoch + 1:>4}/{self.num_rounds}"
                    f"  total={avg_loss:.4f}"
                    f"  bce={avg_bce:.4f}"
                    f"  {self.ranking_loss}={avg_rank:.4f}"
                    + margin_str
                )

        self.feature_names_ = feature_names
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        import torch

        dev   = self._get_device()
        X_arr = (X.values if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        X_arr = self.scaler_.transform(X_arr).astype(np.float32)
        X_t   = torch.tensor(X_arr, device=dev)

        self.net_.eval()
        with torch.no_grad():
            score, _, _, _ = self.net_(X_t)
        out = score.cpu().numpy()
        if np.isnan(out).any():
            import warnings
            warnings.warn(
                "[MLPRanker] predict() returned NaN (training diverged). "
                "Try reducing lr or increasing weight_decay. Returning zeros.",
                RuntimeWarning, stacklevel=2,
            )
            out = np.zeros_like(out)
        return out

    # ------------------------------------------------------------------
    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        return {}

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        import torch, pickle
        torch.save({
            "net_state_dict": self.net_.state_dict(),
            "scaler": pickle.dumps(self.scaler_),
            "config": {
                "input_dim":  self.net_.block1[0].in_features,
                "hidden_dim": self.hidden_dim,
                "dropout":    self.dropout,
            },
        }, filepath)

    def load_model(self, filepath: str) -> "MultiScaleMLPRanker":
        import torch, pickle
        ckpt = torch.load(filepath, map_location="cpu")
        cfg  = ckpt["config"]
        self.net_ = _build_net(cfg["input_dim"], cfg["hidden_dim"], cfg["dropout"])
        self.net_.load_state_dict(ckpt["net_state_dict"])
        if "scaler" in ckpt:
            self.scaler_ = pickle.loads(ckpt["scaler"])
        self.net_.to(self._get_device())
        return self
