"""
policy_network.py
=================
Custom Stable-Baselines3 policy components for the monthly portfolio env.

Architecture Overview
---------------------
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SharedWeightExtractor                        │
    │                                                                 │
    │  For each stock slot i ∈ [0, max_N):                            │
    │      h_i = MLP_shared( features_i )       ← same weights ∀ i   │
    │                                                                 │
    │  Aggregate = [ mean(h), max(h), w_cash ]  ← set-level summary  │
    │  Output    = concat( h_0, h_1, …, h_{N-1}, Aggregate )         │
    └─────────────────────────────────────────────────────────────────┘

    The Actor head produces raw logits → masked softmax → portfolio weights.
    The Critic head produces a scalar V(s).

    Because every stock is processed by the **same** MLP encoder, the
    architecture is invariant to permutation of stocks within each slot,
    and gracefully handles variable-size universes via action masking.

Why not MaskablePPO?
---------------------
    sb3-contrib's MaskablePPO only supports **Discrete** / **MultiDiscrete**
    action spaces.  Our portfolio weights are continuous (softmax output).
    Instead, we implement masking *inside* the policy network: the actor
    applies −∞ to invalid logits before softmax.  The result is equivalent
    to MaskablePPO's masking but works with continuous actions.

    If you want a discrete formulation (e.g. MultiDiscrete bins), swap in
    MaskablePPO and use the ``DiscretizedPortfolioWrapper`` provided at the
    bottom of this file.

Algo Flexibility
-----------------
    The ``SharedWeightExtractor`` is registered as a custom features_extractor
    and works with PPO, SAC, and TD3 out of the box.  Switch the algo in
    config.yaml → policy.algo.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


# ═══════════════════════════════════════════════════════════════════════════
#  Shared-Weight Feature Extractor (Set Transformer Lite)
# ═══════════════════════════════════════════════════════════════════════════

class SharedWeightExtractor(BaseFeaturesExtractor):
    """
    Processes the observation dict from MonthlyPortfolioEnv.

    Input
    -----
    obs dict with:
        "obs"         : (batch, state_dim)
        "action_mask" : (batch, max_N + 1)

    Output
    ------
    features : (batch, features_dim)
        Concatenation of per-stock embeddings + set-level aggregation.

    The ``features_dim`` property tells SB3 how wide the output is.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        n_features: int,
        max_universe_size: int,
        encoder_hidden_dims: list[int] = [64, 64],
        activation: str = "ReLU",
    ):
        # We compute features_dim before calling super().__init__
        encoder_out_dim = encoder_hidden_dims[-1]
        agg_dim = encoder_out_dim * 2 + 1          # mean + max + cash_weight
        _features_dim = max_universe_size * encoder_out_dim + agg_dim

        super().__init__(observation_space, features_dim=_features_dim)

        self.n_features        = n_features
        self.max_N             = max_universe_size
        self.encoder_out_dim   = encoder_out_dim

        # ---- per-stock shared-weight MLP ---
        # Input: n_features (z-scored) + 1 (current portfolio weight for this stock)
        act_cls = getattr(nn, activation, nn.ReLU)
        layers: list[nn.Module] = []
        in_dim = n_features + 1               # +1 for per-stock weight w_{t-1}
        for h in encoder_hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        self.stock_encoder = nn.Sequential(*layers)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs  = observations["obs"]           # (B, state_dim)
        mask = observations["action_mask"]   # (B, max_N + 1)

        B = obs.shape[0]
        n_f = self.n_features
        max_N = self.max_N

        # ---- unpack observation ---
        # Layout:  [features(max_N × n_f), weights(max_N + 1)]
        feat_flat = obs[:, : max_N * n_f]                     # (B, max_N*n_f)
        weight_vec = obs[:, max_N * n_f :]                    # (B, max_N + 1)

        feat_3d = feat_flat.reshape(B, max_N, n_f)            # (B, max_N, n_f)

        # ---- concatenate per-stock weight w_{t-1} with features ---
        # This lets the encoder see what the agent currently owns in each
        # stock, enabling turnover-aware decisions.
        #
        # Scale weight by max_N so that equal-weight (1/N) maps to ≈ 1.0,
        # comparable to z-scored features in [-3, +3].  Without this,
        # a typical weight ≈ 0.0007 is 4 orders of magnitude smaller
        # than the features and gets ignored by the encoder early in
        # training.
        stock_weights = weight_vec[:, :max_N].unsqueeze(-1)   # (B, max_N, 1)
        stock_weights_scaled = stock_weights * max_N           # EW → ~1.0
        encoder_in = torch.cat([feat_3d, stock_weights_scaled], dim=-1)  # (B, max_N, n_f+1)

        # ---- shared encoder per stock ---
        # Reshape to (B * max_N, n_f+1), encode, reshape back
        flat_in  = encoder_in.reshape(B * max_N, n_f + 1)
        flat_out = self.stock_encoder(flat_in)                # (B*max_N, enc_dim)
        h = flat_out.reshape(B, max_N, self.encoder_out_dim)  # (B, max_N, enc_dim)

        # ---- zero out masked stocks for aggregation ---
        stock_mask = mask[:, :-1].unsqueeze(-1)               # (B, max_N, 1)
        h_masked = h * stock_mask                             # zero inactive

        # ---- set-level aggregation ---
        n_active = stock_mask.sum(dim=1).clamp(min=1)         # (B, 1)
        h_mean = h_masked.sum(dim=1) / n_active               # (B, enc_dim)
        h_max, _ = h_masked.max(dim=1)                        # (B, enc_dim)
        cash_w = weight_vec[:, -1:]                           # (B, 1)

        agg = torch.cat([h_mean, h_max, cash_w], dim=-1)     # (B, agg_dim)

        # ---- concatenate per-stock embeddings + aggregation ---
        h_flat = h.reshape(B, max_N * self.encoder_out_dim)
        out = torch.cat([h_flat, agg], dim=-1)                # (B, features_dim)

        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Masked-Softmax Actor Head
# ═══════════════════════════════════════════════════════════════════════════
#
#  SB3's ActorCriticPolicy uses Gaussian distributions for continuous
#  actions.  Our "actions" are portfolio weights produced via softmax,
#  so we wrap the standard policy to intercept the action and apply
#  masking + softmax in a post-processing step.
#
#  The training loop sees raw Gaussian actions; the env's step()
#  applies masked_softmax() to convert them to valid weights.
#  This decouples the policy from the masking logic and keeps SB3
#  internals (log-prob, entropy) consistent.
# ═══════════════════════════════════════════════════════════════════════════


def make_policy_kwargs(
    n_features: int,
    max_universe_size: int,
    encoder_hidden_dims: list[int],
    actor_hidden_dims: list[int],
    critic_hidden_dims: list[int],
    activation: str = "ReLU",
) -> dict:
    """
    Build the ``policy_kwargs`` dict to pass to SB3's PPO / SAC / TD3.

    Usage
    -----
        policy_kwargs = make_policy_kwargs(...)
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
    """
    return dict(
        features_extractor_class=SharedWeightExtractor,
        features_extractor_kwargs=dict(
            n_features=n_features,
            max_universe_size=max_universe_size,
            encoder_hidden_dims=encoder_hidden_dims,
            activation=activation,
        ),
        net_arch=dict(
            pi=actor_hidden_dims,
            vf=critic_hidden_dims,
        ),
        activation_fn=getattr(nn, activation, nn.ReLU),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Optional: Discretized Wrapper for MaskablePPO
# ═══════════════════════════════════════════════════════════════════════════

class DiscretizedPortfolioWrapper(gym.ActionWrapper):
    """
    Wraps the continuous MonthlyPortfolioEnv into a MultiDiscrete env
    compatible with sb3-contrib's MaskablePPO.

    Each stock slot gets ``n_bins`` possible allocation levels [0..n_bins-1].
    The wrapper normalises the chosen bins into portfolio weights.

    Action Mask (for MaskablePPO):
        For each of the (max_N + 1) slots:
            • Active slot:  all n_bins bins are valid → [1,1,…,1]
            • Masked slot:  only bin 0 is valid       → [1,0,…,0]
        Flattened to shape ((max_N + 1) * n_bins,).

    Usage
    -----
        from sb3_contrib import MaskablePPO

        base_env  = MonthlyPortfolioEnv(interp, cfg)
        disc_env  = DiscretizedPortfolioWrapper(base_env, n_bins=21)
        model     = MaskablePPO("MultiInputPolicy", disc_env)
    """

    def __init__(self, env: gym.Env, n_bins: int = 21):
        super().__init__(env)
        self.n_bins = n_bins
        mask_dim = env.observation_space["action_mask"].shape[0]
        self.n_slots = mask_dim   # max_N + 1

        # Override action space to MultiDiscrete
        self.action_space = gym.spaces.MultiDiscrete(
            [n_bins] * self.n_slots
        )

        # Override observation space to include the expanded mask
        obs_space = env.observation_space
        self.observation_space = gym.spaces.Dict({
            "obs":         obs_space["obs"],
            "action_mask": gym.spaces.MultiBinary(self.n_slots * n_bins),
        })

    def action(self, act: np.ndarray) -> np.ndarray:
        """Convert discrete bins → continuous logits for the base env."""
        # The base env expects raw logits; we convert bin indices to
        # proportional values and let the env's masked_softmax handle the rest.
        logits = act.astype(np.float32)
        return logits

    def observation(self, obs: dict) -> dict:
        """Expand the per-slot binary mask → per-slot × per-bin mask."""
        slot_mask = obs["action_mask"]           # (n_slots,)
        expanded  = np.zeros(self.n_slots * self.n_bins, dtype=np.int8)

        for i in range(self.n_slots):
            start = i * self.n_bins
            if slot_mask[i] > 0.5:
                # Active: all bins valid
                expanded[start : start + self.n_bins] = 1
            else:
                # Masked: only bin 0 (= zero allocation) valid
                expanded[start] = 1

        obs_out = dict(obs)
        obs_out["action_mask"] = expanded
        return obs_out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        cont_action = self.action(action)
        obs, rew, term, trunc, info = self.env.step(cont_action)
        return self.observation(obs), rew, term, trunc, info