import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ContinualMetricsEvaluator:
    """
    Computes formal Continual Learning metrics for intrusion detection research:
    - Plasticity (New Attack F1, Precision, Recall)
    - Stability (Known Macro-F1, Retention)
    - Average Catastrophic Forgetting
    - Maximum Class Forgetting
    - Memory Reduction Ratio (MRR)
    - Plasticity-Stability Index (PSI)
    - Edge IoT Viability Index (EIVI)
    """
    
    @staticmethod
    def evaluate_adaptation(
        model_before,
        model_after,
        X_test_known: pd.DataFrame,
        y_test_known: np.ndarray,
        X_eval_new_attack: pd.DataFrame,
        y_eval_new_attack: np.ndarray,
        known_classes: List[int],
        new_class_id: int,
        training_time: float,
        replay_samples: int,
        full_train_samples: int
    ) -> Dict[str, Any]:
        """
        Evaluates a model transition from before to after adaptation.
        """
        # 1. Performance before on known classes
        y_pred_before = model_before.predict(X_test_known)
        f1_before_per_class = f1_score(
            y_test_known, y_pred_before, average=None, labels=known_classes, zero_division=0
        )
        macro_f1_before = f1_score(y_test_known, y_pred_before, average='macro', zero_division=0)
        
        # 2. Performance after on known classes
        y_pred_after = model_after.predict(X_test_known)
        acc_after = accuracy_score(y_test_known, y_pred_after)
        macro_f1_after = f1_score(y_test_known, y_pred_after, average='macro', zero_division=0)
        weighted_f1_after = f1_score(y_test_known, y_pred_after, average='weighted', zero_division=0)
        f1_after_per_class = f1_score(
            y_test_known, y_pred_after, average=None, labels=known_classes, zero_division=0
        )
        
        # 3. Catastrophic Forgetting Metrics
        forgetting_per_class = []
        for idx in range(len(known_classes)):
            # Positive drop means forgetting occurred
            drop = float(f1_before_per_class[idx] - f1_after_per_class[idx])
            forgetting_per_class.append(drop)
            
        avg_forgetting = float(np.mean(forgetting_per_class))
        max_forgetting = float(np.max(forgetting_per_class))
        
        # Backward Transfer (BWT) = - avg_forgetting
        bwt = - avg_forgetting
        
        # 4. Plasticity on New Attack
        y_pred_new = model_after.predict(X_eval_new_attack)
        new_attack_acc = accuracy_score(y_eval_new_attack, y_pred_new)
        new_attack_prec = precision_score(
            y_eval_new_attack, y_pred_new, average='macro', labels=[new_class_id], zero_division=0
        )
        new_attack_rec = recall_score(
            y_eval_new_attack, y_pred_new, average='macro', labels=[new_class_id], zero_division=0
        )
        new_attack_f1 = f1_score(
            y_eval_new_attack, y_pred_new, average='macro', labels=[new_class_id], zero_division=0
        )
        
        # 5. Resource Efficiency Metrics
        memory_reduction = 100.0 * (1.0 - (replay_samples / max(1, full_train_samples)))
        
        # Plasticity-Stability Index: Harmonic mean of Plasticity (New F1) and Stability (1 - Avg Forgetting)
        stability_score = max(0.0, 1.0 - max(0.0, avg_forgetting))
        if (new_attack_f1 + stability_score) > 0:
            psi = 2 * (new_attack_f1 * stability_score) / (new_attack_f1 + stability_score)
        else:
            psi = 0.0
            
        # Edge IoT Viability Index (0 to 100 composite index):
        # Combines High Plasticity (30%), High Stability (30%), Extreme Memory Reduction (30%), Fast Latency (10%)
        latency_score = min(1.0, 1.0 / max(0.01, training_time))
        eivi = (
            (0.35 * new_attack_f1) +
            (0.35 * (1.0 - max(0.0, avg_forgetting))) +
            (0.25 * (memory_reduction / 100.0)) +
            (0.05 * min(1.0, latency_score / 10.0))
        ) * 100.0

        return {
            'known_accuracy': acc_after,
            'known_macro_f1_before': macro_f1_before,
            'known_macro_f1_after': macro_f1_after,
            'known_weighted_f1_after': weighted_f1_after,
            'average_forgetting': avg_forgetting,
            'maximum_forgetting': max_forgetting,
            'backward_transfer': bwt,
            'new_attack_accuracy': new_attack_acc,
            'new_attack_precision': new_attack_prec,
            'new_attack_recall': new_attack_rec,
            'new_attack_f1': new_attack_f1,
            'replay_samples': replay_samples,
            'full_train_samples': full_train_samples,
            'replay_reduction_percent': memory_reduction,
            'training_time_seconds': training_time,
            'plasticity_stability_index': psi,
            'edge_iot_viability_index': eivi
        }
