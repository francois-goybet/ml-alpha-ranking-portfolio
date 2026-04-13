"""GRU-based multi-horizon ranking model.

Architecture
------------
1. Encoder MLP   : F → hidden_dim  (produces initial GRU hidden state h0)
2. GRU cell unrolled 3 steps (same input x fed at every step):
     h1 = GRU(x, h0) → aux_head_1  (binary: top-10% at t+1?)
     h2 = GRU(x, h1) → aux_head_2  (binary: top-10% at t+2?)
     h3 = GRU(x, h2) → aux_head_3  (binary: top-10% at t+3?)
3. Ranking MLP   : h3 → scalar ranking score  (used at inference only)

Losses
------
  α · BCE(aux1, top10_t1) + β · BCE(aux2, top10_t2) + γ · BCE(aux3, top10_t3)
  + δ · RankingLoss(score, rank_labels)

Default weights: α=1.0, β=0.7, γ=0.4, δ=1.0

Ranking losses (controlled by ``ranking_loss`` parameter)
---------------------------------------------------------
"lambda" (default) — LambdaLoss:
  Per-group pairwise loss weighted by |ΔNDCG|:
    weight_ij = |ΔNDCG_ij|  (gain from swapping positions in the current ranking)
    loss_ij   = weight_ij · log(1 + exp(−σ · sign(lbl_i−lbl_j) · (s_i−s_j)))

"ndcg" — NeuralNDCG (Pobrotyn & Bartczak 2021):
  Approximates NDCG using a differentiable soft-sort (NeuralSort):
    P_hat = NeuralSort(scores, tau)         — soft permutation matrix (N×N)
    dcg   = gains @ P_hat @ discounts       — differentiable DCG
    loss  = 1 − dcg / idcg                  — minimise (1 − NDCG)
  Controlled by ``ndcg_tau`` (temperature; lower = harder sort, default 1.0).

At inference only the ranking head score is returned; auxiliary heads are discarded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.model.model import BaseRankingModel, _LABEL_ENCODERS


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _GRUNet:
    """Lazily imported so torch is not required at module import time."""
    pass


def _build_gru_net(input_dim: int, hidden_dim: int, encoder_layers: int,
                   ranking_head_layers: int, dropout: float):
    import torch.nn as nn

    # --- Encoder MLP: input_dim → hidden_dim ---
    enc: list = []
    in_d = input_dim
    for _ in range(encoder_layers):
        enc += [nn.Linear(in_d, hidden_dim), nn.ReLU()]
        if dropout > 0:
            enc.append(nn.Dropout(dropout))
        in_d = hidden_dim
    encoder = nn.Sequential(*enc)

    # --- GRU cell (processes each stock independently across 3 horizons) ---
    # Input: raw features (input_dim); hidden state: hidden_dim
    gru_cell = nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim)

    # --- Auxiliary binary heads (one per horizon) ---
    aux1 = nn.Linear(hidden_dim, 1)
    aux2 = nn.Linear(hidden_dim, 1)
    aux3 = nn.Linear(hidden_dim, 1)

    # --- Ranking MLP: hidden_dim → 1 ---
    rank_layers: list = []
    in_r = hidden_dim
    for _ in range(ranking_head_layers):
        rank_layers += [nn.Linear(in_r, hidden_dim), nn.ReLU()]
        if dropout > 0:
            rank_layers.append(nn.Dropout(dropout))
        in_r = hidden_dim
    rank_layers.append(nn.Linear(in_r, 1))
    ranking_head = nn.Sequential(*rank_layers)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.gru_cell = gru_cell
            self.aux_head_1 = aux1
            self.aux_head_2 = aux2
            self.aux_head_3 = aux3
            self.ranking_head = ranking_head

        def forward(self, x):
            # x: (N, F)
            h = self.encoder(x)                        # h0: (N, hidden_dim)
            h1 = self.gru_cell(x, h)                   # (N, hidden_dim)
            h2 = self.gru_cell(x, h1)                  # (N, hidden_dim)
            h3 = self.gru_cell(x, h2)                  # (N, hidden_dim)

            a1 = self.aux_head_1(h1).squeeze(-1)       # (N,)
            a2 = self.aux_head_2(h2).squeeze(-1)       # (N,)
            a3 = self.aux_head_3(h3).squeeze(-1)       # (N,)
            score = self.ranking_head(h3).squeeze(-1)  # (N,)
            return score, a1, a2, a3

    return Net()


# ---------------------------------------------------------------------------
# LambdaLoss (per group)
# ---------------------------------------------------------------------------

def _lambda_loss_group(scores, labels, sigma: float = 1.0):
    """LambdaLoss for a single ranking group.

    Parameters
    ----------
    scores : torch.Tensor, shape (N,)
    labels : torch.Tensor, shape (N,)  — non-negative integer relevance
    sigma  : float  — scaling of the logistic pair loss

    Returns
    -------
    torch.Tensor  — scalar loss
    """
    import torch

    N = len(scores)
    if N < 2:
        return scores.sum() * 0.0

    device = scores.device
    labels = labels.float()

    # Ideal DCG
    sorted_labels, _ = torch.sort(labels, descending=True)
    positions = torch.arange(1, N + 1, dtype=torch.float32, device=device)
    gains_ideal = 2.0 ** sorted_labels - 1.0
    discounts = 1.0 / torch.log2(positions + 1.0)
    idcg = (gains_ideal * discounts).sum()

    if idcg < 1e-10:
        return scores.sum() * 0.0

    # Current rank (1-based) from predicted scores
    sorted_idx = torch.argsort(scores, descending=True)
    rank = torch.empty(N, dtype=torch.float32, device=device)
    rank[sorted_idx] = positions

    # All distinct pairs (i, j)  — upper triangle only
    rows, cols = torch.triu_indices(N, N, offset=1)
    rows, cols = rows.to(device), cols.to(device)

    lbl_r = labels[rows]
    lbl_c = labels[cols]

    # Keep only pairs with different labels
    mask = lbl_r != lbl_c
    if mask.sum() == 0:
        return scores.sum() * 0.0

    rows, cols = rows[mask], cols[mask]
    lbl_r, lbl_c = lbl_r[mask], lbl_c[mask]

    rank_r = rank[rows]
    rank_c = rank[cols]

    gain_r = 2.0 ** lbl_r - 1.0
    gain_c = 2.0 ** lbl_c - 1.0
    disc_r = 1.0 / torch.log2(rank_r + 1.0)
    disc_c = 1.0 / torch.log2(rank_c + 1.0)

    # |ΔNDCG| if items r and c were swapped in the ranking
    delta_ndcg = torch.abs((gain_r - gain_c) * (disc_c - disc_r)) / idcg

    # direction: +1 when lbl_r > lbl_c (we want s_r > s_c)
    direction = torch.sign(lbl_r - lbl_c)
    score_diff = scores[rows] - scores[cols]

    pair_loss = delta_ndcg * torch.log1p(torch.exp(-sigma * direction * score_diff))
    return pair_loss.sum()


# ---------------------------------------------------------------------------
# ListNet top-1 loss (per group)  — Cao et al. 2007, O(N)
# ---------------------------------------------------------------------------

def _listnet_loss_group(scores, labels, tau: float = 1.0):
    """ListNet top-1 cross-entropy loss for a single ranking group.

    Parameters
    ----------
    scores : torch.Tensor, shape (N,)
    labels : torch.Tensor, shape (N,)  — non-negative relevance grades
    tau    : float  — softmax temperature (higher = softer distribution)

    Returns
    -------
    torch.Tensor  — scalar loss
    """
    import torch.nn.functional as F
    if len(scores) < 2:
        return scores.sum() * 0.0
    p_pred = F.log_softmax(scores / tau, dim=0)
    p_true = F.softmax(labels.float() / tau, dim=0)
    return -(p_true * p_pred).sum()


# ---------------------------------------------------------------------------
# NeuralNDCG loss (per group)
# Differentiable NDCG via NeuralSort (Pobrotyn & Bartczak, 2021).
# ---------------------------------------------------------------------------

def _neural_ndcg_loss_group(scores, labels, tau: float = 1.0):
    """NeuralNDCG loss for a single ranking group.

    Parameters
    ----------
    scores : torch.Tensor, shape (N,)
    labels : torch.Tensor, shape (N,)  — non-negative relevance grades
    tau    : float  — NeuralSort temperature; lower → harder (sharper) sort

    Returns
    -------
    torch.Tensor  — scalar loss  (1 − approx_NDCG)
    """
    import torch

    N = len(scores)
    if N < 2:
        return scores.sum() * 0.0

    device = scores.device
    labels = labels.float()

    # Ideal DCG
    sorted_labels, _ = torch.sort(labels, descending=True)
    positions = torch.arange(1, N + 1, dtype=torch.float32, device=device)
    gains_ideal = 2.0 ** sorted_labels - 1.0
    discounts = 1.0 / torch.log2(positions + 1.0)
    idcg = (gains_ideal * discounts).sum()
    if idcg < 1e-10:
        return scores.sum() * 0.0

    # NeuralSort soft permutation matrix  P_hat[i,j] ≈ Prob(item i is at position j)
    #
    # Formula (Grover et al. 2019, eq. 7):
    #   logit[i, j] = (N+1 - 2j) * s_i  -  sum_k |s_i - s_k|
    #   P_hat[:, j] = softmax( logit[:, j] / tau )      ← over items i (dim=0)
    #
    # softmax over items per position (dim=0) → each column (position) sums to 1,
    # i.e. each position receives exactly one item's worth of gain → DCG ≤ IDCG ≤ 1.
    # (dim=1 / row-softmax would let multiple items share a position → DCG > IDCG, wrong.)
    positions = torch.arange(1, N + 1, dtype=torch.float32, device=device)  # (N,)
    A_s = torch.abs(scores.unsqueeze(1) - scores.unsqueeze(0))  # (N, N) pairwise |s_i - s_j|
    B   = A_s.sum(dim=1)                                         # (N,)   sum_k |s_i - s_k|
    scaling = (N + 1 - 2 * positions)                           # (N,)   position-dependent weight
    # logits[i, j] = scaling[j] * s_i  -  B[i]
    logits  = scores.unsqueeze(1) * scaling.unsqueeze(0) - B.unsqueeze(1)  # (N, N)
    P_hat   = torch.softmax(logits / tau, dim=0)                # (N, N) cols sum to 1

    # DCG: gains (N,) @ P_hat (N,N) @ discounts (N,)   → scalar
    gains = 2.0 ** labels - 1.0                                    # (N,)
    dcg = (gains.unsqueeze(0) @ P_hat @ discounts.unsqueeze(1)).squeeze()
    ndcg = dcg / idcg
    return 1.0 - ndcg


# ---------------------------------------------------------------------------
# GRURanker — BaseRankingModel-compatible
# ---------------------------------------------------------------------------

class GRURanker(BaseRankingModel):
    """GRU multi-horizon ranker with LambdaLoss + BCE auxiliary heads.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of encoder MLP and GRU state.
    encoder_layers : int
        Number of hidden layers in the encoder MLP.
    ranking_head_layers : int
        Number of hidden layers in the final ranking MLP.
    dropout : float
        Dropout rate applied after each hidden layer.
    alpha, beta_aux, gamma : float
        BCE loss weights for auxiliary heads at t+1, t+2, t+3.
    delta : float
        LambdaLoss weight.
    lambda_sigma : float
        σ parameter inside the LambdaLoss logistic term.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation for Adam.
    num_rounds : int
        Number of training epochs (reuses BaseRankingModel's ``num_rounds``).
    batch_groups : int
        Number of monthly groups per gradient step.
    device : str
        ``"cpu"``, ``"cuda"``, or ``"mps"``.
    label_encoder : str
        Name of the rank-label encoder used for LambdaLoss targets.
        Default ``"long_ranker"`` (top-10% linear labels, rest 0).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        encoder_layers: int = 2,
        ranking_head_layers: int = 2,
        dropout: float = 0.1,
        alpha: float = 1.0,
        beta_aux: float = 0.7,
        gamma: float = 0.4,
        delta: float = 1.0,
        lambda_sigma: float = 1.0,
        ranking_loss: str = "lambda",
        ndcg_tau: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 0.0,
        num_rounds: int = 20,
        batch_groups: int = 8,
        device: str = "cpu",
        label_encoder: str = "long_ranker",
        # Forwarded to BaseRankingModel (unused by this model but kept for API compat)
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
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.ranking_head_layers = ranking_head_layers
        self.dropout = dropout
        self.alpha = alpha
        self.beta_aux = beta_aux
        self.gamma = gamma
        self.delta = delta
        self.lambda_sigma = lambda_sigma
        self.ranking_loss = ranking_loss
        self.ndcg_tau = ndcg_tau
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = max_grad_norm
        self.batch_groups = batch_groups
        self.device = device
        self.label_encoder = label_encoder
        self._input_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Property override — net_ instead of model_
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "net_")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_device(self):
        import torch
        if self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _build_binary_targets(
        y: pd.DataFrame,
        groups: List[int],
        col_idx: int,
    ) -> np.ndarray:
        """Return 1 for top-10% per month, 0 otherwise."""
        out = np.zeros(len(y), dtype=np.float32)
        vals = y.iloc[:, col_idx].values
        cursor = 0
        for g in groups:
            sl = slice(cursor, cursor + g)
            top_k = max(1, g // 10)
            top_idx = np.argsort(vals[sl])[::-1][:top_k]
            out[cursor + top_idx] = 1.0
            cursor += g
        return out

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | pd.Series,
        groups: List[int] | np.ndarray,
        eval_set: Optional[tuple] = None,     # unused — kept for API compat
        eval_groups: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> "GRURanker":
        import torch
        import torch.nn as nn

        dev = self._get_device()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        X_arr = (X.values if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Standard-scale features: zero mean, unit variance per feature (fit on train only).
        # This prevents large-magnitude inputs from causing gradient explosion in the GRU.
        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler()
        X_arr = self.scaler_.fit_transform(X_arr).astype(np.float32)

        groups = list(groups)

        # --- Rank labels for LambdaLoss (primary target = first column / series) ---
        if isinstance(y, pd.DataFrame):
            y_primary = y.iloc[:, 0]
        else:
            y_primary = y if isinstance(y, pd.Series) else pd.Series(y)

        encoder_fn = _LABEL_ENCODERS.get(self.label_encoder, _LABEL_ENCODERS["long_ranker"])
        rank_labels = encoder_fn(y_primary, groups).astype(np.float32)

        # --- Binary targets for auxiliary BCE heads ---
        if isinstance(y, pd.DataFrame) and y.shape[1] >= 3:
            bin1 = self._build_binary_targets(y, groups, 0)  # t+1
            bin2 = self._build_binary_targets(y, groups, 1)  # t+2
            bin3 = self._build_binary_targets(y, groups, 2)  # t+3
        else:
            # Fallback: derive binary targets from the primary series
            y_df = pd.DataFrame({"_": y_primary.values})
            bin1 = self._build_binary_targets(y_df, groups, 0)
            bin2, bin3 = bin1.copy(), bin1.copy()

        # --- Move everything to device tensors ---
        X_t = torch.tensor(X_arr, device=dev)
        rank_t = torch.tensor(rank_labels, device=dev)
        bin1_t = torch.tensor(bin1, device=dev)
        bin2_t = torch.tensor(bin2, device=dev)
        bin3_t = torch.tensor(bin3, device=dev)

        # --- Build network ---
        input_dim = X_arr.shape[1]
        self._input_dim = input_dim
        self.net_ = _build_gru_net(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            encoder_layers=self.encoder_layers,
            ranking_head_layers=self.ranking_head_layers,
            dropout=self.dropout,
        ).to(dev)

        optimizer = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        bce_fn = nn.BCEWithLogitsLoss()

        # Pre-compute group start offsets
        group_starts = np.concatenate([[0], np.cumsum(groups[:-1])])
        num_groups = len(groups)

        if self.ranking_loss not in ("lambda", "ndcg", "softmax"):
            raise ValueError(f"ranking_loss must be 'lambda', 'ndcg', or 'softmax', got '{self.ranking_loss}'")

        self.training_history_: Dict[str, list] = {
            "loss": [], "bce": [], "rank": []
        }

        log_every = max(1, self.num_rounds // min(10, self.num_rounds))

        for epoch in range(self.num_rounds):
            self.net_.train()
            perm = np.random.permutation(num_groups)
            epoch_loss = 0.0
            epoch_bce = 0.0
            epoch_rank = 0.0
            n_batches = 0

            for batch_start in range(0, num_groups, self.batch_groups):
                batch_idxs = perm[batch_start: batch_start + self.batch_groups]

                # Gather contiguous row indices for this batch of groups
                row_indices = np.concatenate([
                    np.arange(group_starts[gi], group_starts[gi] + groups[gi])
                    for gi in batch_idxs
                ])
                row_t = torch.tensor(row_indices, dtype=torch.long, device=dev)

                x_b      = X_t[row_t]
                rank_b   = rank_t[row_t]
                b1, b2, b3 = bin1_t[row_t], bin2_t[row_t], bin3_t[row_t]
                batch_group_sizes = [groups[gi] for gi in batch_idxs]

                score, a1, a2, a3 = self.net_(x_b)

                # BCE on auxiliary heads
                loss_bce = (
                    self.alpha    * bce_fn(a1, b1)
                    + self.beta_aux * bce_fn(a2, b2)
                    + self.gamma    * bce_fn(a3, b3)
                )

                # Ranking loss (LambdaLoss or NeuralNDCG) averaged over batch
                rank_total = torch.tensor(0.0, device=dev)
                cursor = 0
                for g_size in batch_group_sizes:
                    sl = slice(cursor, cursor + g_size)
                    if self.ranking_loss == "ndcg":
                        rank_total = rank_total + _neural_ndcg_loss_group(
                            score[sl], rank_b[sl], tau=self.ndcg_tau
                        )
                    elif self.ranking_loss == "softmax":
                        rank_total = rank_total + _listnet_loss_group(
                            score[sl], rank_b[sl], tau=self.ndcg_tau
                        )
                    else:
                        rank_total = rank_total + _lambda_loss_group(
                            score[sl], rank_b[sl], sigma=self.lambda_sigma
                        )
                    cursor += g_size
                rank_avg = rank_total / len(batch_group_sizes)

                total_loss = loss_bce + self.delta * rank_avg

                optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    import torch.nn as _nn
                    _nn.utils.clip_grad_norm_(self.net_.parameters(), self.max_grad_norm)
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_bce  += loss_bce.item()
                epoch_rank += rank_avg.item()
                n_batches += 1

            nb = max(1, n_batches)
            avg_loss = epoch_loss / nb
            avg_bce  = epoch_bce  / nb
            avg_rank = epoch_rank / nb
            self.training_history_["loss"].append(avg_loss)
            self.training_history_["bce"].append(avg_bce)
            self.training_history_["rank"].append(avg_rank)

            if verbose and (epoch % log_every == 0 or epoch == self.num_rounds - 1):
                print(
                    f"  [GRURanker] epoch {epoch + 1:>4}/{self.num_rounds}"
                    f"  total={avg_loss:.4f}"
                    f"  bce={avg_bce:.4f}"
                    f"  {self.ranking_loss}={avg_rank:.4f}"
                )

        self.feature_names_ = feature_names
        return self

    # ------------------------------------------------------------------
    # predict  — only ranking head score is used
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        import torch

        dev = self._get_device()
        X_arr = (X.values if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        X_arr = self.scaler_.transform(X_arr).astype(np.float32)
        X_t = torch.tensor(X_arr, device=dev)

        self.net_.eval()
        with torch.no_grad():
            score, _, _, _ = self.net_(X_t)
        out = score.cpu().numpy()
        if np.isnan(out).any():
            # Fallback to zeros if training diverged; avoids downstream crashes.
            import warnings
            warnings.warn(
                "[GRURanker] predict() returned NaN scores (training diverged). "
                "Try reducing lr or increasing weight_decay. Returning zeros.",
                RuntimeWarning,
                stacklevel=2,
            )
            out = np.zeros_like(out)
        return out

    # ------------------------------------------------------------------
    # Misc BaseRankingModel methods
    # ------------------------------------------------------------------

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        # Tree-based feature importance is not applicable to GRU models.
        return {}

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        import torch
        import pickle
        torch.save(
            {
                "net_state_dict": self.net_.state_dict(),
                "scaler": pickle.dumps(self.scaler_),
                "config": {
                    "input_dim": self._input_dim,
                    "hidden_dim": self.hidden_dim,
                    "encoder_layers": self.encoder_layers,
                    "ranking_head_layers": self.ranking_head_layers,
                    "dropout": self.dropout,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str) -> "GRURanker":
        import torch, pickle
        ckpt = torch.load(filepath, map_location="cpu")
        cfg = ckpt["config"]
        if "scaler" in ckpt:
            self.scaler_ = pickle.loads(ckpt["scaler"])
        self.net_ = _build_gru_net(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            encoder_layers=cfg["encoder_layers"],
            ranking_head_layers=cfg["ranking_head_layers"],
            dropout=cfg["dropout"],
        )
        self.net_.load_state_dict(ckpt["net_state_dict"])
        self._input_dim = cfg["input_dim"]
        dev = self._get_device()
        self.net_.to(dev)
        return self
