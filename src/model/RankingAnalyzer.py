from typing import Optional
import numpy as np
import pandas as pd
from src.model.model import MultiHorizonRanker, HorizonEnsemble
from sklearn.metrics import roc_auc_score

from src.visualization.model_plots import plot_history

"""RankingAnalyzer for evaluating multi-horizon ranking models."""

class RankingAnalyzer:
    """
    Analyzes predictions from a MultiHorizonRanker and HorizonEnsemble.
    
    Args:
        model: A fitted MultiHorizonRanker instance.
        ensemble: A fitted HorizonEnsemble instance.
        X_test: Test feature matrix (DataFrame).
        group_test: List of stock counts per month in test set.
        y_test: Test target DataFrame with columns matching model.targets.
    """
    
    def __init__(
        self,
        model: MultiHorizonRanker,
        ensemble: HorizonEnsemble,
        X_test: pd.DataFrame,
        group_test: list[int],
        y_test: pd.DataFrame,
    ):
        if not model.is_fitted:
            raise ValueError("model must be fitted before creating RankingAnalyzer.")
        if not ensemble.multi_ranker.is_fitted:
            raise ValueError("ensemble's multi_ranker must be fitted.")
        
        self.model = model
        self.ensemble = ensemble
        self.X_test = X_test
        self.group_test = group_test
        self.y_test = y_test
        
        # Validate shapes
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have the same length.")
        if sum(group_test) != len(X_test):
            raise ValueError("sum(group_test) must equal len(X_test).")
    
    def evaluate(self, eval_at: list[int], encoder_fn: Optional[callable] = None) -> pd.DataFrame:
        """
        Evaluates both the multi-horizon ranker and the ensemble.
        Returns a DataFrame with metrics for each target and the ensemble.
        """
        df_multi = self.evaluate_multi_horizon_ranker(eval_at, encoder_fn)
        df_ensemble = self.evaluate_ensemble(eval_at, encoder_fn)
        return pd.concat([df_multi, df_ensemble], ignore_index=True)

    def evaluate_multi_horizon_ranker(self, eval_at: list[int], encoder_fn: Optional[callable] = None) -> pd.DataFrame:
        """
        Evaluates model and ensemble predictions using AUC, NDCG@k, and HIT@k.
        Args:
            eval_at: List of k values for NDCG@k and HIT@k evaluation.
            encoder_fn: Optional function to encode continuous targets to integer grades.
        Returns:
            DataFrame with evaluation metrics for each target and the ensemble.
        """
        rows = []
        predictions = self.model.predict(self.X_test)

        for target, scores in predictions.items():
            y_t = self.y_test[target]
            groups = list(self.group_test)

            encoded = encoder_fn(y_t, groups) if encoder_fn else y_t.to_numpy()

            ndcg_scores = {k: [] for k in eval_at}
            hit_scores = {k: [] for k in eval_at}

            cursor = 0
            for g in groups:
                sl = slice(cursor, cursor + g)
                for k in eval_at:
                    ndcg_scores[k].append(self.ndcg_at_k(scores[sl], encoded[sl].astype(float), k))
                    hit_scores[k].append(self.precision_at_k(scores[sl], encoded[sl].astype(float), k))
                cursor += g

            auc = roc_auc_score((encoded > np.median(encoded)).astype(int), scores)

            row = {
                "target": target,
                "AUC": auc,
            }

            for k in eval_at:
                row[f"NDCG@{k}"] = np.mean(ndcg_scores[k])
                row[f"HIT@{k}"] = np.mean(hit_scores[k])

            rows.append(row)

        return pd.DataFrame(rows)
    
    def evaluate_ensemble(self, eval_at: list[int], encoder_fn: Optional[callable] = None) -> pd.DataFrame:
        """
        Evaluates ensemble predictions using AUC, NDCG@k, and HIT@k.
        Args:
            eval_at: List of k values for NDCG@k and HIT@k evaluation.
            encoder_fn: Optional function to encode continuous targets to integer grades.   
        Returns:
            DataFrame with evaluation metrics for the ensemble.
        """

        ensemble_scores = self.ensemble.predict(self.X_test, groups=list(self.group_test))


        ref_target = list(self.y_test.columns)[0]
        y_ref = self.y_test[ref_target]
        ref_encoded = encoder_fn(y_ref, list(self.group_test)) if encoder_fn else y_ref.to_numpy()

        ndcg_scores = {k: [] for k in eval_at}
        hit_scores = {k: [] for k in eval_at}

        cursor = 0
        for g in self.group_test:
            sl = slice(cursor, cursor + g)
            for k in eval_at:
                ndcg_scores[k].append(self.ndcg_at_k(ensemble_scores[sl], ref_encoded[sl].astype(float), k))
                hit_scores[k].append(self.precision_at_k(ensemble_scores[sl], ref_encoded[sl].astype(float), k))
            cursor += g

        auc_ens = roc_auc_score((ref_encoded > np.median(ref_encoded)).astype(int), ensemble_scores)

        row = {
            "target": f"ensemble/{self.ensemble.combination}",
            "AUC": auc_ens,
        }

        for k in eval_at:
            row[f"NDCG@{k}"] = np.mean(ndcg_scores[k])
            row[f"HIT@{k}"] = np.mean(hit_scores[k])

        return pd.DataFrame([row])
    
    def get_history_figures(self):
        history = self.model.get_history()
        figs = plot_history(history)
        return history, figs

    def ndcg_at_k(self, scores, labels, k):
        order = np.argsort(scores)[::-1][:k]
        gains = labels[order]
        discounts = np.log2(np.arange(2, len(gains) + 2))
        dcg = np.sum(gains / discounts)

        ideal_order = np.argsort(labels)[::-1][:k]
        ideal_gains = labels[ideal_order]
        idcg = np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2)))

        return dcg / idcg if idcg > 0 else 0.0

    def precision_at_k(self, scores, labels, k):
        k = min(k, len(scores))
        top_k_pred = set(np.argsort(scores)[::-1][:k])
        top_k_actual = set(np.argsort(labels)[::-1][:k])
        return len(top_k_pred & top_k_actual) / k