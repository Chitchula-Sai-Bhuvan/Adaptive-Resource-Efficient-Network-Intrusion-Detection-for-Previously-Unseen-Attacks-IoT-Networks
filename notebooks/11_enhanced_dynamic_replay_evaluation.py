import os
import sys
sys.path.insert(0, '/Users/saibhuvan/Documents/Research/imple')
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.continual.dynamic_replay import DynamicContinualLearner
from src.evaluation.continual_metrics import ContinualMetricsEvaluator

os.chdir('/Users/saibhuvan/Documents/Research/imple')
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/verification", exist_ok=True)

print("Starting Enhanced Multi-Attack Continual Learning Evaluation...")

# 1. Load Data
df_raw = pd.read_csv("data/raw/train_test_network.csv")
df_raw = df_raw.replace('-', pd.NA)
df_raw = df_raw.drop(columns=['label', 'src_ip', 'dst_ip'], errors='ignore')
df_raw = df_raw.drop_duplicates()

attacks_to_test = ['mitm', 'ransomware', 'xss', 'backdoor']
all_results = []

for unseen_attack in attacks_to_test:
    print(f"\n========================================================")
    print(f"Benchmarking Unseen Attack: {unseen_attack.upper()}")
    print(f"========================================================")
    
    # Isolate unseen class vs known classes
    unseen_df = df_raw[df_raw['type'] == unseen_attack]
    known_df = df_raw[df_raw['type'] != unseen_attack]
    
    X_known = known_df.drop(columns=['type'])
    y_known = known_df['type']
    X_unseen = unseen_df.drop(columns=['type'])
    y_unseen = unseen_df['type']
    
    X_train, X_test_known, y_train, y_test_known = train_test_split(
        X_known, y_known, test_size=0.3, random_state=42, stratify=y_known
    )
    
    # Preprocess
    valid_cols = X_train.columns[X_train.isna().sum() < len(X_train) * 0.5]
    X_train = X_train[valid_cols].copy()
    X_test_known = X_test_known[valid_cols].copy()
    X_unseen = X_unseen[valid_cols].copy()
    
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
    
    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test_known[col] = X_test_known[col].fillna(med)
        X_unseen[col] = X_unseen[col].fillna(med)
        
    for col in cat_cols:
        mode = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode)
        X_test_known[col] = X_test_known[col].fillna(mode)
        X_unseen[col] = X_unseen[col].fillna(mode)
        
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(X_train[cat_cols])
        
        def encode(data):
            enc = ohe.transform(data[cat_cols])
            enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(cat_cols), index=data.index)
            return pd.concat([data.drop(columns=cat_cols), enc_df], axis=1)
            
        X_train = encode(X_train)
        X_test_known = encode(X_test_known)
        X_unseen = encode(X_unseen)
        
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_known_enc = le.transform(y_test_known)
    
    unseen_id = len(le.classes_)
    y_unseen_enc = np.full(len(y_unseen), unseen_id)
    known_labels_list = list(range(len(le.classes_)))
    full_train_size = len(X_train)
    
    # Train Initial Model (Baseline)
    learner = DynamicContinualLearner(random_state=42)
    learner.fit_initial(X_train, y_train_enc)
    model_baseline = learner.model
    
    # Split unseen attack into 20% adaptation and 80% untouched evaluation
    X_unseen_adapt, X_unseen_eval, y_unseen_adapt, y_unseen_eval = train_test_split(
        X_unseen, y_unseen_enc, test_size=0.8, random_state=42
    )
    
    # Evaluate Baseline (No adaptation)
    res_baseline = ContinualMetricsEvaluator.evaluate_adaptation(
        model_before=model_baseline,
        model_after=model_baseline,
        X_test_known=X_test_known,
        y_test_known=y_test_known_enc,
        X_eval_new_attack=X_unseen_eval,
        y_eval_new_attack=y_unseen_eval,
        known_classes=known_labels_list,
        new_class_id=unseen_id,
        training_time=0.0,
        replay_samples=full_train_size,
        full_train_samples=full_train_size
    )
    res_baseline.update({
        'unseen_attack': unseen_attack,
        'strategy': 'Static Baseline (No Adaptation)',
        'adaptation_samples': 0,
        'quota_per_class': 0
    })
    all_results.append(res_baseline)
    
    # Strategies to compare
    strategies_to_run = [
        ('full', 'Full Retraining / Full Replay', 0),
        ('static', 'Static Replay (100/cls)', 100),
        ('dynamic_volume', 'Dynamic Volume Replay', 0),
        ('dynamic_smart', 'Optimized Smart Dynamic Replay', 0)
    ]
    
    for strat_key, strat_display, static_param in strategies_to_run:
        print(f" -> Running Strategy: {strat_display}...")
        adapted_model, meta = learner.adapt_with_replay(
            X_train_full=X_train,
            y_train_full=y_train_enc,
            X_new_attack=X_unseen_adapt,
            y_new_attack=y_unseen_adapt,
            strategy=strat_key,
            static_quota=static_param
        )
        
        eval_metrics = ContinualMetricsEvaluator.evaluate_adaptation(
            model_before=model_baseline,
            model_after=adapted_model,
            X_test_known=X_test_known,
            y_test_known=y_test_known_enc,
            X_eval_new_attack=X_unseen_eval,
            y_eval_new_attack=y_unseen_eval,
            known_classes=known_labels_list,
            new_class_id=unseen_id,
            training_time=meta['training_time'],
            replay_samples=meta['replay_samples'],
            full_train_samples=full_train_size
        )
        eval_metrics.update({
            'unseen_attack': unseen_attack,
            'strategy': strat_display,
            'adaptation_samples': meta['incoming_attack_samples'],
            'quota_per_class': meta['quota_per_class']
        })
        all_results.append(eval_metrics)

# Compile results
df_all = pd.DataFrame(all_results)
df_all.to_csv("results/tables/master_multi_attack_comprehensive.csv", index=False)
print("Saved master_multi_attack_comprehensive.csv successfully!")

# Summary comparison table focusing on key trade-offs
cols_summary = [
    'unseen_attack', 'strategy', 'replay_samples', 'replay_reduction_percent',
    'new_attack_f1', 'known_macro_f1_after', 'average_forgetting', 'maximum_forgetting',
    'plasticity_stability_index', 'edge_iot_viability_index', 'training_time_seconds'
]
df_summary = df_all[cols_summary].copy()
df_summary.to_csv("results/tables/master_iot_edge_tradeoff_metrics.csv", index=False)
print("Saved master_iot_edge_tradeoff_metrics.csv successfully!")
