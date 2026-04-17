#!/usr/bin/env python3
"""
debug.py — Visualise state & action snapshots saved by StateActionLoggerCallback.

Usage:
    python debug.py                              # default: episode 500
    python debug.py --episode 1000               # specific episode
    python debug.py --log-dir artifacts/logs     # custom log directory
    python debug.py --save dashboard_ep500.png   # save to specific path
    python debug.py --config config.yaml         # read feature names from config
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
#  Defaults — must match config.yaml / train.py
# ---------------------------------------------------------------------------
MAX_N = 2200

DEFAULT_FEATURE_NAMES = [
    "xgboost_rank_oos",
    "DolVol", "BidAskSpread", "RealizedVol",
    "VolMkt", "TrendFactor",
    "Size", "BM", "Mom12m",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def resolve_feature_names(npz_data, config_path):
    """Get feature column names: .npz → config.yaml → hardcoded defaults."""
    if "feature_cols" in npz_data and len(npz_data["feature_cols"]) > 0:
        names = list(npz_data["feature_cols"])
        if names and names[0]:
            return names

    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        base = list(cfg["features"]["base_features"])
        prefix = []
        for key in ["optional_alpha", "optional_meta"]:
            col = cfg["features"].get(key)
            if col:
                prefix.append(col)
        return prefix + base

    return list(DEFAULT_FEATURE_NAMES)


def load_snapshot(log_dir, episode):
    path = Path(log_dir) / f"state_action_ep{episode}.npz"
    if not path.exists():
        available = sorted(Path(log_dir).glob("state_action_ep*.npz"))
        if available:
            eps = [int(p.stem.split("ep")[1]) for p in available]
            print(f"Episode {episode} not found.  Available: {eps}")
        else:
            print(f"No snapshot files in {log_dir}/")
        sys.exit(1)
    return np.load(path, allow_pickle=True)


def parse_obs(obs, action_mask, n_features):
    """
    Split the flat observation vector.

    Observation layout (from MonthlyPortfolioEnv / MonthlyStateInterpreter):
        obs[0 : MAX_N * n_features]   → per-stock features  (MAX_N, n_features)
        obs[MAX_N * n_features : ]    → portfolio weights    (MAX_N + 1,)
                                         last element = cash weight
    """
    feat_end = MAX_N * n_features
    features = obs[:feat_end].reshape(MAX_N, n_features)
    weights  = obs[feat_end:]                      # (MAX_N + 1,)
    return features, weights


def detect_active_stocks(features, action_mask):
    """
    Determine which stock slots are active.

    Primary:  use action_mask from observation (if the env populates it).
    Fallback: infer from features — a slot with ANY non-zero feature is active.
              (Padded slots are all-zero; z-scored active stocks are almost
              never exactly zero across all features simultaneously.)
    """
    mask_stocks = action_mask[:-1]                 # exclude cash slot
    n_from_mask = int((mask_stocks > 0.5).sum())

    if n_from_mask > 0:
        return mask_stocks > 0.5, "action_mask"

    # Fallback: any row with at least one non-zero feature
    row_abs_sum = np.abs(features).sum(axis=1)     # (MAX_N,)
    inferred = row_abs_sum > 1e-8
    return inferred, "inferred_from_features"


# ═══════════════════════════════════════════════════════════════════════════
#  Text summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(features, weights, active_mask, action, feature_names,
                  detection_source, episode):
    n_active    = int(active_mask.sum())
    n_features  = features.shape[1]
    cash_weight = weights[-1]
    cash_logit  = action[-1]

    print("\n" + "=" * 65)
    print(f"  Episode {episode} Snapshot")
    print("=" * 65)

    print(f"\n  Universe        : {n_active} active stocks  (padded to {MAX_N})")
    print(f"  Active detected : via {detection_source}")
    print(f"  State features  : {n_features} columns")
    print(f"  Obs vector len  : {len(features.ravel()) + len(weights)}")

    if n_active == 0:
        print("\n  ⚠  No active stocks detected (even from features).")
        print("     The observation may be from a degenerate state.")
        return False

    af = features[active_mask]
    aw = weights[:MAX_N][active_mask]
    aa = action[:MAX_N][active_mask]

    # ── Feature table ──
    print(f"\n  {'─'*61}")
    print(f"  {'Feature':<22s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}  {'NaN':>5s}")
    print(f"  {'─'*61}")
    for i, name in enumerate(feature_names[:n_features]):
        col = af[:, i]
        nan_count = int(np.isnan(col).sum())
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            print(f"  {name:<22s} {valid.mean():>8.4f} {valid.std():>8.4f} "
                  f"{valid.min():>8.4f} {valid.max():>8.4f}  {nan_count:>5d}")
        else:
            print(f"  {name:<22s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s}  {nan_count:>5d}")

    # ── Weights (in state) ──
    print(f"\n  Portfolio weights (in state):")
    print(f"    Stock weights : sum={aw.sum():.6f}  min={aw.min():.6f}  "
          f"max={aw.max():.6f}  nonzero={int((aw > 1e-8).sum())}")
    print(f"    Cash weight   : {cash_weight:.6f}")
    print(f"    Total         : {aw.sum() + cash_weight:.6f}")

    # ── Actions (policy output) ──
    print(f"\n  Action logits (policy output):")
    print(f"    Stock logits  : mean={aa.mean():.4f}  std={aa.std():.4f}  "
          f"min={aa.min():.4f}  max={aa.max():.4f}")
    print(f"    Cash logit    : {cash_logit:.4f}")

    # ── Top / bottom ──
    order = np.argsort(aw)[::-1]
    n_show = min(5, n_active)
    print(f"\n  Top-{n_show} stock weights : {aw[order[:n_show]]}")
    print(f"  Bottom-{n_show} stock weights: {aw[order[-n_show:]]}")

    # ── Softmax preview (what the env would produce from these logits) ──
    full_mask = np.zeros(MAX_N + 1, dtype=np.float32)
    full_mask[:MAX_N][active_mask] = 1.0
    full_mask[-1] = 1.0
    masked = np.where(full_mask > 0.5, action, -1e9)
    shifted = masked - masked.max()
    exp_vals = np.exp(shifted)
    sm_weights = exp_vals / exp_vals.sum()
    sm_stocks = sm_weights[:MAX_N][active_mask]
    sm_cash = sm_weights[-1]
    print(f"\n  Softmax weights (from logits):")
    print(f"    Stock range   : [{sm_stocks.min():.6f}, {sm_stocks.max():.6f}]  "
          f"sum={sm_stocks.sum():.6f}")
    print(f"    Cash          : {sm_cash:.6f}")
    print("=" * 65)
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Dashboard
# ═══════════════════════════════════════════════════════════════════════════

def plot_dashboard(features, weights, active_mask, action, feature_names,
                   detection_source, episode, save_path=None):
    n_active = int(active_mask.sum())
    n_feat   = features.shape[1]
    af = features[active_mask]
    aw = weights[:MAX_N][active_mask]
    aa = action[:MAX_N][active_mask]
    cash_w = weights[-1]
    cash_a = action[-1]

    # Compute softmax weights
    full_mask = np.zeros(MAX_N + 1, dtype=np.float32)
    full_mask[:MAX_N][active_mask] = 1.0
    full_mask[-1] = 1.0
    masked = np.where(full_mask > 0.5, action, -1e9)
    shifted = masked - masked.max()
    exp_vals = np.exp(shifted)
    sm_all = exp_vals / exp_vals.sum()
    sm_stocks = sm_all[:MAX_N][active_mask]
    sm_cash = sm_all[-1]

    fig = plt.figure(figsize=(20, 14), facecolor="white")
    fig.suptitle(f"Episode {episode}  ·  {n_active} active stocks  ·  "
                 f"{n_feat} features  ·  mask: {detection_source}",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 4, hspace=0.40, wspace=0.32,
                           left=0.05, right=0.97, top=0.94, bottom=0.05)
    labels = feature_names[:n_feat]

    # ── Row 1, Col 1: State weights distribution ────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    nz = aw[aw > 1e-8]
    ax.hist(nz if len(nz) else aw, bins=50, color="#4C72B0",
            edgecolor="white", linewidth=0.3)
    eq = 1.0 / n_active
    ax.axvline(eq, color="red", ls="--", lw=1, label=f"EW = {eq:.5f}")
    ax.set_title(f"State Weights ({len(nz)} non-zero)")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # ── Row 1, Col 2: Action logit distribution ─────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(aa, bins=50, color="#55A868", edgecolor="white", linewidth=0.3)
    ax.axvline(cash_a, color="red", ls="--", lw=1,
               label=f"Cash logit = {cash_a:.3f}")
    ax.set_title("Action Logits (active)")
    ax.set_xlabel("Logit")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # ── Row 1, Col 3: Softmax weights distribution ─────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(sm_stocks, bins=50, color="#C44E52", edgecolor="white", linewidth=0.3)
    ax.axvline(sm_cash, color="blue", ls="--", lw=1,
               label=f"Cash = {sm_cash:.4f}")
    ax.axvline(eq, color="green", ls=":", lw=1, label=f"EW = {eq:.5f}")
    ax.set_title("Softmax Weights (from logits)")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # ── Row 1, Col 4: Budget bar chart ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    stock_sum_state = aw.sum()
    stock_sum_sm = sm_stocks.sum()
    x = np.arange(2)
    w_bar = 0.35
    ax.bar(x - w_bar/2, [stock_sum_state, cash_w], w_bar,
           label="In state", color=["#4C72B0", "#DD8452"])
    ax.bar(x + w_bar/2, [stock_sum_sm, sm_cash], w_bar,
           label="From logits", color=["#7CB5D2", "#F0B87A"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Stocks", "Cash"])
    ax.set_ylabel("Weight")
    ax.set_title("Weight Budget")
    ax.legend(fontsize=7)

    # ── Row 2, Col 1-2: Feature box-plots ───────────────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    bp = ax.boxplot(af, vert=True, patch_artist=True, showfliers=False,
                    widths=0.6)
    cmap = plt.cm.tab10(np.linspace(0, 1, n_feat))
    for patch, c in zip(bp["boxes"], cmap):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, n_feat + 1))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_title("Feature Distributions (active stocks, z-scored)")
    ax.set_ylabel("Value")
    ax.axhline(0, color="grey", ls=":", lw=0.8)

    # ── Row 2, Col 3-4: Feature heatmap ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2:4])
    sample_n = min(100, n_active)
    idx = np.sort(np.random.choice(n_active, sample_n, replace=False))
    im = ax.imshow(af[idx].T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Stock index (sample of {sample_n})")
    ax.set_title("Feature Heatmap")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # ── Row 3, Col 1: Weight vs alpha feature ───────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(af[:, 0], aw, s=3, alpha=0.35, color="#C44E52")
    ax.set_xlabel(labels[0])
    ax.set_ylabel("State weight")
    ax.set_title(f"State Weight vs {labels[0]}")

    # ── Row 3, Col 2: Softmax weight vs alpha feature ───────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(af[:, 0], sm_stocks, s=3, alpha=0.35, color="#8172B2")
    ax.set_xlabel(labels[0])
    ax.set_ylabel("Softmax weight")
    ax.set_title(f"Softmax Weight vs {labels[0]}")

    # ── Row 3, Col 3: Logit vs alpha feature ────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(af[:, 0], aa, s=3, alpha=0.35, color="#55A868")
    ax.set_xlabel(labels[0])
    ax.set_ylabel("Action logit")
    ax.set_title(f"Logit vs {labels[0]}")

    # ── Row 3, Col 4: State weight vs Action logit ──────────────────────
    ax = fig.add_subplot(gs[2, 3])
    ax.scatter(aa, aw, s=3, alpha=0.35, color="#DD8452")
    ax.set_xlabel("Action logit")
    ax.set_ylabel("State weight")
    ax.set_title("State Weight vs Logit")

    out = save_path or f"dashboard_ep{episode}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Dashboard saved → {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualise state/action snapshots from RL training")
    parser.add_argument("--episode",  type=int, default=500)
    parser.add_argument("--log-dir",  default="artifacts/logs")
    parser.add_argument("--config",   default="config.yaml",
                        help="Path to config.yaml for feature column names")
    parser.add_argument("--save",     default=None,
                        help="Save figure path (default: dashboard_ep<N>.png)")
    args = parser.parse_args()

    # ── Load snapshot ─────────────────────────────────────────────────
    npz = load_snapshot(args.log_dir, args.episode)
    obs         = npz["obs"]
    action_mask = npz["action_mask"]
    action      = npz["action"]

    # ── Resolve feature names ─────────────────────────────────────────
    feature_names = resolve_feature_names(npz, args.config)
    n_features    = len(feature_names)

    # Sanity-check obs length and auto-correct n_features if needed
    expected_len = MAX_N * n_features + MAX_N + 1
    if len(obs) != expected_len:
        inferred = (len(obs) - MAX_N - 1) // MAX_N
        print(f"  ⚠  obs length {len(obs)} != expected {expected_len} "
              f"(n_features={n_features})")
        print(f"     Inferred n_features = {inferred} from obs length.")
        n_features = inferred
        if len(feature_names) != n_features:
            feature_names = [f"feat_{i}" for i in range(n_features)]

    # ── Parse ─────────────────────────────────────────────────────────
    features, weights = parse_obs(obs, action_mask, n_features)
    active_mask, detection_source = detect_active_stocks(features, action_mask)

    # ── Print + Plot ──────────────────────────────────────────────────
    has_active = print_summary(features, weights, active_mask, action,
                               feature_names, detection_source, args.episode)
    if has_active:
        plot_dashboard(features, weights, active_mask, action,
                       feature_names, detection_source, args.episode,
                       save_path=args.save)


if __name__ == "__main__":
    main()