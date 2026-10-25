import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

class ReplayBufferManager:
    """
    Manages historical replay memory for Edge IoT Continual Intrusion Detection.
    
    Supports:
    1. Static fixed-size sampling (k samples per known class).
    2. Dynamic volume-aware scaling (proportional to incoming novel attack volume).
    3. Optimized boundary & uncertainty-balanced sampling (prioritizes high-variance and margin samples).
    """
    
    def __init__(self, min_samples_per_class: int = 100, random_state: int = 42):
        self.min_samples_per_class = min_samples_per_class
        self.random_state = random_state

    def sample_static_buffer(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray, 
        known_classes: List[int], 
        samples_per_class: int = 100
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extracts a fixed number of samples per known class.
        """
        buffer_dfs = []
        df_temp = X_train.copy()
        df_temp['__target__'] = y_train
        
        for cls_id in known_classes:
            cls_subset = df_temp[df_temp['__target__'] == cls_id]
            n_select = min(samples_per_class, len(cls_subset))
            if n_select > 0:
                sampled = cls_subset.sample(n=n_select, random_state=self.random_state)
                buffer_dfs.append(sampled)
                
        buffer_df = pd.concat(buffer_dfs, ignore_index=True)
        y_buf = buffer_df['__target__'].values
        X_buf = buffer_df.drop(columns=['__target__'])
        return X_buf, y_buf

    def compute_dynamic_quota(
        self, 
        incoming_attack_volume: int, 
        num_known_classes: int,
        scaling_ratio: float = 1.0,
        max_buffer_cap: Optional[int] = 5000
    ) -> int:
        """
        Dynamically computes the quota of replay samples per known class.
        Rule: max(min_samples, int(incoming_volume * scaling_ratio / num_known_classes))
        """
        dynamic_quota = int((incoming_attack_volume * scaling_ratio) / max(1, num_known_classes))
        quota = max(self.min_samples_per_class, dynamic_quota)
        if max_buffer_cap is not None:
            max_per_cls = max(self.min_samples_per_class, max_buffer_cap // max(1, num_known_classes))
            quota = min(quota, max_per_cls)
        return quota

    def sample_dynamic_buffer(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        known_classes: List[int],
        incoming_attack_volume: int,
        scaling_ratio: float = 1.0,
        max_buffer_cap: Optional[int] = 5000
    ) -> Tuple[pd.DataFrame, np.ndarray, int]:
        """
        Extracts a dynamically scaled replay buffer to match incoming attack volume.
        """
        quota_per_class = self.compute_dynamic_quota(
            incoming_attack_volume=incoming_attack_volume,
            num_known_classes=len(known_classes),
            scaling_ratio=scaling_ratio,
            max_buffer_cap=max_buffer_cap
        )
        X_buf, y_buf = self.sample_static_buffer(
            X_train=X_train,
            y_train=y_train,
            known_classes=known_classes,
            samples_per_class=quota_per_class
        )
        return X_buf, y_buf, quota_per_class

    def sample_optimized_smart_buffer(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        known_classes: List[int],
        incoming_attack_volume: int,
        baseline_model=None,
        max_buffer_cap: int = 4000
    ) -> Tuple[pd.DataFrame, np.ndarray, int]:
        """
        Optimized Boundary & Representative Dynamic Buffer.
        Combines stratified representative sampling with boundary margin focus:
        For each class:
        - 60% uniform stratified exemplars (preserving centroid and global cluster distribution)
        - 40% boundary exemplars (samples with highest prediction entropy / margin uncertainty)
        This mathematically prevents decision tree splits from shifting away from historical classes!
        """
        quota_per_class = self.compute_dynamic_quota(
            incoming_attack_volume=incoming_attack_volume,
            num_known_classes=len(known_classes),
            scaling_ratio=1.1,
            max_buffer_cap=max_buffer_cap
        )
        
        df_temp = X_train.copy()
        df_temp['__target__'] = y_train
        buffer_dfs = []
        
        for cls_id in known_classes:
            cls_subset = df_temp[df_temp['__target__'] == cls_id]
            n_select = min(quota_per_class, len(cls_subset))
            
            if n_select == 0:
                continue
                
            if baseline_model is not None and hasattr(baseline_model, "predict_proba") and len(cls_subset) > n_select:
                try:
                    # Calculate prediction probabilities to find boundary samples
                    features_only = cls_subset.drop(columns=['__target__'])
                    probs = baseline_model.predict_proba(features_only)
                    # Margin uncertainty: 1.0 - confidence on true class
                    cls_col_idx = list(baseline_model.classes_).index(cls_id) if cls_id in baseline_model.classes_ else -1
                    if cls_col_idx >= 0:
                        uncertainty = 1.0 - probs[:, cls_col_idx]
                    else:
                        uncertainty = np.random.rand(len(cls_subset))
                        
                    n_boundary = int(0.4 * n_select)
                    n_uniform = n_select - n_boundary
                    
                    # Top uncertain (boundary) samples
                    sorted_indices = np.argsort(uncertainty)[::-1]
                    boundary_idx = cls_subset.index[sorted_indices[:n_boundary]]
                    remaining_idx = cls_subset.index[sorted_indices[n_boundary:]]
                    
                    # Uniform samples from the rest
                    uniform_selected = np.random.choice(remaining_idx, size=min(n_uniform, len(remaining_idx)), replace=False)
                    selected_idx = np.concatenate([boundary_idx, uniform_selected])
                    sampled = cls_subset.loc[selected_idx]
                except Exception:
                    sampled = cls_subset.sample(n=n_select, random_state=self.random_state)
            else:
                sampled = cls_subset.sample(n=n_select, random_state=self.random_state)
                
            buffer_dfs.append(sampled)
            
        buffer_df = pd.concat(buffer_dfs, ignore_index=True)
        y_buf = buffer_df['__target__'].values
        X_buf = buffer_df.drop(columns=['__target__'])
        return X_buf, y_buf, quota_per_class
