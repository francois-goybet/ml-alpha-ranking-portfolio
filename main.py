import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config.config_loader import load_config

from src.data.DataManager import DataManager
from src.model.model import MultiHorizonRanker, HorizonEnsemble, _LABEL_ENCODERS

def main(args):

    # Load configuration with lightweight parsing.
    config = load_config(args.config)
    data_manager = DataManager(config.get("data", {}))
    data_manager.get_data(start=config["data"].get("train_start", "1990-01-01"), end=config["data"].get("test_end", "2024-12-31"), market_cap=config["data"].get("market_cap", 10))
    s = data_manager.get_train_val_test(targets=["ret_1m", "ret_3m", "ret_6m"])
    # print stats about the splits
    for split_name, (X, y, group) in s.items():

        print(f"{split_name}: {len(X)} samples, {len(X.columns)} features, {len(set(group))} groups.")
        # print columns of y
        print(f"  Targets: {y.columns.tolist()}")
        # print number of nan values in y
        print(f"  NaN values in targets: {y.isna().sum().to_dict()}")
        # print firsts columns of X
        print(f"  First 5 columns of X: {X.columns[:5].tolist()}")
        # print group first 5 values
        print(f"  First 5 group values: {group[:5]}")
        # print size of the first group
        print(f"  Size of first group: {group[0]}")
        print()
    X_train, y_train, group_train = s["train"]
    X_val, y_val, group_val = s["val"]
    X_test, y_test, group_test = s["test"]

    # Check if some values in y_train are NaN (should not happen with current data loading, but just in case).
    if y_train.isna().any().any():
        print("Warning: NaN values found in y_train. This may cause issues during training.")
        print(y_train.isna().sum())

    targets = config.get("model", {}).get("targets", ["ret_1m", "ret_3m", "ret_6m"])
    model = MultiHorizonRanker(targets=targets, **config.get("model", {}))
    model.fit(X_train, y_train, group_train, (X_val, y_val), group_val, verbose=True)

    # --- Post-training evaluation on test set ---
    label_encoder_name = config.get("model", {}).get("label_encoder", "argsort")
    eval_at = config.get("model", {}).get("eval_at", [10, 20])
    encoder_fn = _LABEL_ENCODERS.get(label_encoder_name)

    print("\n--- Test set evaluation ---")
    predictions = model.predict(X_test)
    for target, scores in predictions.items():
        y_t = y_test[target]
        groups = list(group_test)

        # NDCG@k per group, then average
        def ndcg_at_k(scores, labels, k):
            order = np.argsort(scores)[::-1][:k]
            gains = labels[order]
            discounts = np.log2(np.arange(2, len(gains) + 2))
            dcg = np.sum(gains / discounts)
            ideal_order = np.argsort(labels)[::-1][:k]
            ideal_gains = labels[ideal_order]
            idcg = np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2)))
            return dcg / idcg if idcg > 0 else 0.0

        def precision_at_k(scores, labels, k):
            """Fraction of top-k predicted stocks that are in the top-k actual stocks."""
            k = min(k, len(scores))
            top_k_pred = set(np.argsort(scores)[::-1][:k])
            top_k_actual = set(np.argsort(labels)[::-1][:k])
            return len(top_k_pred & top_k_actual) / k

        encoded = encoder_fn(y_t, groups) if encoder_fn else y_t.to_numpy()
        ndcg_scores = {k: [] for k in eval_at}
        hit_scores = {k: [] for k in eval_at}
        cursor = 0
        for g in groups:
            sl = slice(cursor, cursor + g)
            for k in eval_at:
                ndcg_scores[k].append(ndcg_at_k(scores[sl], encoded[sl].astype(float), k))
                hit_scores[k].append(precision_at_k(scores[sl], encoded[sl].astype(float), k))
            cursor += g

        auc = roc_auc_score((encoded > np.median(encoded)).astype(int), scores)
        ndcg_parts = "  ".join(f"NDCG@{k}: {np.mean(ndcg_scores[k]):.4f}" for k in eval_at)
        hit_parts = "  ".join(f"Hit@{k}: {np.mean(hit_scores[k]):.4f}" for k in eval_at)
        print(f"  [{target}]  {ndcg_parts}  {hit_parts}  AUC: {auc:.4f}")

    # --- Ensemble evaluation ---
    ensemble_cfg = config.get("ensemble", {})
    if len(targets) > 1 and ensemble_cfg.get("enabled", True):
        weights = ensemble_cfg.get("weights", None)
        combination = ensemble_cfg.get("combination", "mean_rank")
        ensemble = HorizonEnsemble(model, combination=combination, weights=weights)
        ensemble_scores = ensemble.predict(X_test, groups=list(group_test))

        # Evaluate ensemble against ret_1m as reference target
        ref_target = targets[0]
        y_ref = y_test[ref_target]
        ref_encoded = encoder_fn(y_ref, list(group_test)) if encoder_fn else y_ref.to_numpy()
        ndcg_scores_ens = {k: [] for k in eval_at}
        hit_scores_ens = {k: [] for k in eval_at}
        cursor = 0
        for g in list(group_test):
            sl = slice(cursor, cursor + g)
            for k in eval_at:
                ndcg_scores_ens[k].append(ndcg_at_k(ensemble_scores[sl], ref_encoded[sl].astype(float), k))
                hit_scores_ens[k].append(precision_at_k(ensemble_scores[sl], ref_encoded[sl].astype(float), k))
            cursor += g
        auc_ens = roc_auc_score((ref_encoded > np.median(ref_encoded)).astype(int), ensemble_scores)
        ndcg_parts_ens = "  ".join(f"NDCG@{k}: {np.mean(ndcg_scores_ens[k]):.4f}" for k in eval_at)
        hit_parts_ens = "  ".join(f"Hit@{k}: {np.mean(hit_scores_ens[k]):.4f}" for k in eval_at)
        w_str = str(weights) if weights else "equal"
        print(f"  [ensemble/{combination} w={w_str}]  {ndcg_parts_ens}  {hit_parts_ens}  AUC: {auc_ens:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate AlphaXGBoost model for stock ranking.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_francois.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    
    main(args)
