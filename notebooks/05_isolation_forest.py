# %% [markdown]
# # 05: Anomaly Detection for Previously Unseen Attacks

# %%
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

os.chdir('/Users/saibhuvan/Documents/Research/imple')
os.makedirs("results/figures/phase5", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/verification", exist_ok=True)
os.makedirs("models", exist_ok=True)

# %%
print("Loading data...")
X_train = pd.read_csv("data/processed/X_train_unscaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test = pd.read_csv("data/processed/X_test_unscaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

with open("data/processed/label_mapping.yaml", "r") as f:
    label_mapping = yaml.safe_load(f)

mitm_encoded = label_mapping['mitm']
normal_encoded = label_mapping['normal']

# Sanity checks
assert mitm_encoded not in y_train, "LEAKAGE: MITM found in training set!"
print(f"Known training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %%
print("Training Isolation Forest on known training traffic...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train)

joblib.dump(iso_forest, "models/isolation_forest_baseline.pkl")
print("Saved Isolation Forest to models/isolation_forest_baseline.pkl")

# %%
print("Evaluating anomaly detector...")
# Predict: 1 = inlier, -1 = anomaly
# Convert to: 0 = inlier (normal), 1 = anomaly (attack)
raw_preds = iso_forest.predict(X_test)
y_pred_anomaly = np.where(raw_preds == -1, 1, 0)

# Ground truth binary: 0 = normal, 1 = any attack
y_test_anomaly_gt = np.where(y_test == normal_encoded, 0, 1)

# Masks
mitm_mask = (y_test == mitm_encoded)
normal_mask = (y_test == normal_encoded)
known_attack_mask = (~mitm_mask) & (~normal_mask)

mitm_preds = y_pred_anomaly[mitm_mask]
normal_preds = y_pred_anomaly[normal_mask]
known_attack_preds = y_pred_anomaly[known_attack_mask]

# Calculations
mitm_total = len(mitm_preds)
mitm_detected = int(np.sum(mitm_preds == 1))
mitm_inlier = int(np.sum(mitm_preds == 0))
mitm_det_rate = (mitm_detected / mitm_total) * 100 if mitm_total > 0 else 0

normal_total = len(normal_preds)
normal_fp = int(np.sum(normal_preds == 1))
normal_fp_rate = (normal_fp / normal_total) * 100 if normal_total > 0 else 0

known_attack_total = len(known_attack_preds)
known_attack_detected = int(np.sum(known_attack_preds == 1))
known_attack_det_rate = (known_attack_detected / known_attack_total) * 100 if known_attack_total > 0 else 0

# Precision, Recall, F1 for MITM specifically
# MITM evaluation focuses strictly on identifying MITM samples as anomalous.
mitm_precision = precision_score(y_test_anomaly_gt[mitm_mask], y_pred_anomaly[mitm_mask], pos_label=1, zero_division=0)
mitm_recall = recall_score(y_test_anomaly_gt[mitm_mask], y_pred_anomaly[mitm_mask], pos_label=1, zero_division=0)
mitm_f1 = f1_score(y_test_anomaly_gt[mitm_mask], y_pred_anomaly[mitm_mask], pos_label=1, zero_division=0)

# Confusion Matrix for Isolation Forest (Binary)
cm_binary = confusion_matrix(y_test_anomaly_gt, y_pred_anomaly)

# %%
print("Generating visualizations...")
plt.figure(figsize=(6, 5))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Inlier', 'Predicted Anomaly'],
            yticklabels=['Actual Normal', 'Actual Attack'])
plt.title("Isolation Forest Binary Confusion Matrix")
plt.tight_layout()
plt.savefig("results/figures/phase5/isolation_forest_cm.png")
plt.close()

plt.figure(figsize=(6, 5))
bars = plt.bar(['Detected as Anomaly', 'Classified as Inlier'], [mitm_detected, mitm_inlier], color=['red', 'gray'])
plt.title("MITM Anomaly Detection Results")
plt.ylabel("Number of Samples")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
plt.tight_layout()
plt.savefig("results/figures/phase5/mitm_detection_bar.png")
plt.close()

# %%
print("Building comparison against Random Forest...")
with open("results/metrics/baseline_results.json", "r") as f:
    rf_metrics = json.load(f)

rf_acc = rf_metrics.get('accuracy', 0)
rf_mitm_f1 = rf_metrics.get('f1_mitm', 0)

comparison_data = [
    {
        "Metric": "Known-Class Main Metric",
        "Random Forest": f"Accuracy: {rf_acc:.4f}",
        "Isolation Forest": f"Known Attack Det Rate: {known_attack_det_rate:.2f}%"
    },
    {
        "Metric": "MITM F1 Score",
        "Random Forest": f"{rf_mitm_f1:.4f}",
        "Isolation Forest": f"{mitm_f1:.4f}"
    },
    {
        "Metric": "MITM Detection Rate",
        "Random Forest": "0.00% (Misclassified into known classes)",
        "Isolation Forest": f"{mitm_det_rate:.2f}%"
    }
]
pd.DataFrame(comparison_data).to_csv("results/tables/rf_vs_isolation_forest.csv", index=False)

# Visualizing RF vs ISO comparison for MITM F1
plt.figure(figsize=(6, 5))
bars = plt.bar(['Random Forest', 'Isolation Forest'], [rf_mitm_f1, mitm_f1], color=['blue', 'orange'])
plt.title("MITM Detection F1 Score: RF vs Isolation Forest")
plt.ylabel("F1 Score")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.4f}", va='bottom', ha='center')
plt.tight_layout()
plt.savefig("results/figures/phase5/rf_vs_iso_mitm_f1.png")
plt.close()

# %%
print("Generating final report...")
md = []
md.append("# Phase 5 Report: Anomaly Detection for Previously Unseen Attacks")
md.append("")
md.append("==================================================")
md.append("1. Research objective")
md.append("==================================================")
md.append("The objective is to determine whether an unsupervised anomaly detection model can identify previously unseen MITM network behavior that a closed-set supervised Random Forest classifier fails to recognize. We evaluate a controlled previously-unseen attack class using a class-holdout protocol.")
md.append("")
md.append("==================================================")
md.append("2. Dataset")
md.append("==================================================")
md.append("The validated `train_test_network.csv` dataset, strictly preprocessed and deduplicated to avoid data leakage.")
md.append("")
md.append("==================================================")
md.append("3. Training protocol")
md.append("==================================================")
md.append("The Isolation Forest was fitted **exclusively** on the known-class training partition. No MITM traffic and no attack labels were exposed during the fitting process.")
md.append("")
md.append("==================================================")
md.append("4. Feature representation")
md.append("==================================================")
md.append("The exact same 18-dimension continuous feature representation from the clean Phase 1-4 baseline was utilized (including numeric columns and one-hot encoded metadata, strictly omitting raw IPs and explicit targets).")
md.append("")
md.append("==================================================")
md.append("5. Isolation Forest configuration")
md.append("==================================================")
md.append("`IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)`")
md.append("")
md.append("==================================================")
md.append("6. Training sample count")
md.append("==================================================")
md.append(f"{len(X_train)} samples")
md.append("")
md.append("==================================================")
md.append("7. Test sample count")
md.append("==================================================")
md.append(f"{len(X_test)} samples (Known Normal: {normal_total}, Known Attack: {known_attack_total}, Unseen MITM: {mitm_total})")
md.append("")
md.append("==================================================")
md.append("8. Known normal results")
md.append("==================================================")
md.append(f"Known Normal False-Positive Rate: {normal_fp_rate:.2f}% ({normal_fp}/{normal_total})")
md.append("")
md.append("==================================================")
md.append("9. Known attack results")
md.append("==================================================")
md.append(f"Known Attack Anomaly Detection Rate: {known_attack_det_rate:.2f}% ({known_attack_detected}/{known_attack_total})")
md.append("")
md.append("==================================================")
md.append("10. MITM anomaly detection results")
md.append("==================================================")
md.append(f"- MITM total: {mitm_total}")
md.append(f"- MITM detected as anomaly: {mitm_detected}")
md.append(f"- MITM classified as inlier: {mitm_inlier}")
md.append(f"- MITM anomaly detection rate: {mitm_det_rate:.2f}%")
md.append(f"- MITM precision: {mitm_precision:.4f}")
md.append(f"- MITM recall: {mitm_recall:.4f}")
md.append(f"- MITM F1: {mitm_f1:.4f}")
md.append("")
md.append("==================================================")
md.append("11. Confusion matrix (Overall Binary)")
md.append("==================================================")
md.append(f"True Negative (Actual Normal -> Inlier): {cm_binary[0,0]}")
md.append(f"False Positive (Actual Normal -> Anomaly): {cm_binary[0,1]}")
md.append(f"False Negative (Actual Attack -> Inlier): {cm_binary[1,0]}")
md.append(f"True Positive (Actual Attack -> Anomaly): {cm_binary[1,1]}")
md.append("")
md.append("==================================================")
md.append("12. RF vs Isolation Forest comparison")
md.append("==================================================")
md.append(pd.DataFrame(comparison_data).to_markdown())
md.append("")
md.append("==================================================")
md.append("13. Limitations")
md.append("==================================================")
md.append("The Isolation Forest was deployed directly with `contamination='auto'` without hyperparameter tuning to serve as a pure unsupervised baseline. Results reflect fundamental distributional differences, rather than an optimized defensive boundary.")
md.append("")
md.append("==================================================")
md.append("14. Research interpretation")
md.append("==================================================")
md.append("Random Forest achieves excellent performance on known classes, BUT the MITM class cannot be recognized because it was absent during training. The Isolation Forest, which does not rely on MITM class labels and merely learns the structural distribution of training traffic, flags unseen MITM traffic as structurally anomalous at a distinct rate. A poor anomaly detection rate here verifies that anomaly detection alone is insufficient, motivating adaptive or continual learning methodologies.")
md.append("")
md.append("==================================================")
md.append("15. Conclusion")
md.append("==================================================")
md.append(f"The unsupervised Isolation Forest achieved an anomaly detection rate of {mitm_det_rate:.2f}% against the zero-label MITM attack, producing an F1 score of {mitm_f1:.4f}. When juxtaposed with the supervised Random Forest (MITM F1: {rf_mitm_f1:.4f}), this solidifies the structural gap in defense paradigms when confronted with withheld behavioral classes.")
md.append("")

with open("results/verification/phase5_report.md", "w") as f:
    f.write("\n".join(md))

print("Phase 5 successfully executed.")
