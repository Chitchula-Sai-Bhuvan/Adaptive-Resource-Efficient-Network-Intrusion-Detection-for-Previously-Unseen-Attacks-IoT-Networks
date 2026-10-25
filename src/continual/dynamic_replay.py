import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.continual.buffer_manager import ReplayBufferManager

class DynamicContinualLearner:
    """
    End-to-end Continual Learning Engine for Network Intrusion Detection in IoT Networks.
    Provides automated model updating upon detection of novel attack vectors.
    """
    
    def __init__(
        self,
        base_estimator_params: Optional[Dict[str, Any]] = None,
        min_replay_per_class: int = 100,
        random_state: int = 42
    ):
        self.random_state = random_state
        self.params = base_estimator_params or {
            'n_estimators': 100,
            'max_depth': 20,
            'random_state': random_state,
            'n_jobs': -1
        }
        self.buffer_manager = ReplayBufferManager(
            min_samples_per_class=min_replay_per_class,
            random_state=random_state
        )
        self.model = RandomForestClassifier(**self.params)
        self.classes_seen: List[int] = []

    def fit_initial(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Fits initial base model on historical known network traffic."""
        self.classes_seen = sorted(list(np.unique(y_train)))
        self.model.fit(X_train, y_train)
        return self

    def adapt_with_replay(
        self,
        X_train_full: pd.DataFrame,
        y_train_full: np.ndarray,
        X_new_attack: pd.DataFrame,
        y_new_attack: np.ndarray,
        strategy: str = "dynamic_smart",
        static_quota: int = 100
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Adapts the model to incorporate a newly detected attack vector.
        
        Strategies:
        - 'full': Uses all historical training samples (traditional retraining).
        - 'static': Uses a fixed number of samples per known class.
        - 'dynamic_volume': Scales quota linearly with incoming attack volume.
        - 'dynamic_smart': Uses boundary and representative exemplar allocation.
        """
        incoming_vol = len(X_new_attack)
        known_classes = self.classes_seen
        total_historical = len(X_train_full)
        
        start_time = time.time()
        
        if strategy == "full":
            X_replay = X_train_full
            y_replay = y_train_full
            quota_used = "Full"
        elif strategy == "static":
            X_replay, y_replay = self.buffer_manager.sample_static_buffer(
                X_train_full, y_train_full, known_classes, samples_per_class=static_quota
            )
            quota_used = static_quota
        elif strategy == "dynamic_volume":
            X_replay, y_replay, quota = self.buffer_manager.sample_dynamic_buffer(
                X_train_full, y_train_full, known_classes, incoming_attack_volume=incoming_vol
            )
            quota_used = quota
        elif strategy == "dynamic_smart":
            X_replay, y_replay, quota = self.buffer_manager.sample_optimized_smart_buffer(
                X_train_full, y_train_full, known_classes, 
                incoming_attack_volume=incoming_vol, 
                baseline_model=self.model
            )
            quota_used = quota
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        replay_samples_count = len(X_replay)
        replay_reduction = 100.0 * (1.0 - (replay_samples_count / max(1, total_historical)))
        
        # Merge replay buffer with novel attack samples
        X_combined = pd.concat([X_replay, X_new_attack], ignore_index=True)
        y_combined = np.concatenate([y_replay, y_new_attack])
        
        # Train adapted model
        adapted_model = RandomForestClassifier(**self.params)
        adapted_model.fit(X_combined, y_combined)
        training_time = time.time() - start_time
        
        metadata = {
            'strategy': strategy,
            'quota_per_class': quota_used,
            'replay_samples': replay_samples_count,
            'replay_reduction_percent': replay_reduction,
            'incoming_attack_samples': incoming_vol,
            'training_time': training_time
        }
        
        return adapted_model, metadata
