# %% [markdown]
# # 04: Unseen Attack Experiment

# %%
import pandas as pd
import yaml
import json
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.chdir('..')

# %%
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load test data (which contains both known test samples and unseen attack samples)
X_test = pd.read_csv("data/processed/X_test_unscaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Load the label mapping
with open("data/processed/label_mapping.yaml", "r") as f:
    mapping = yaml.safe_load(f)
    reverse_mapping = {v: k for k, v in mapping.items()}

# Load the trained RF baseline
model = joblib.load("models/random_forest_baseline.pkl")

# %% [markdown]
# ### Evaluation on Test Set with Unseen Attack
# %%
# We evaluate the performance. Notice how the model performs on the unseen attack.
y_pred = model.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Get class names in correct order
classes = [reverse_mapping.get(i, f"Class {i}") for i in range(len(mapping))]

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (Includes Unseen Attack)")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/confusion_matrix.png")
plt.close()

# %% [markdown]
# ### Analysis of Known vs Unseen Attack Performance
# %%
# Load the results from the baseline training
with open("results/metrics/baseline_results.json", "r") as f:
    results = json.load(f)

unseen_attack = config['unseen_attack']

print("--- Final Experimental Result Table (Excerpt) ---")
print(f"Overall Accuracy: {results['accuracy']:.4f}")
print(f"Macro F1: {results['f1_macro']:.4f}")

# Print F1 for each known class
print("\n--- Known Attack F1 Scores ---")
known_f1s = []
for k, v in mapping.items():
    if k != unseen_attack:
        f1 = results.get(f'f1_{k}', 0)
        known_f1s.append(f1)
        print(f"{k}: {f1:.4f}")

# Print F1 for unseen class
unseen_f1 = results.get(f'f1_{unseen_attack}', 0)
print(f"\n--- Unseen Attack F1 Score ---")
print(f"{unseen_attack} (Held-out): {unseen_f1:.4f}")

# %% [markdown]
# ### Visualization of Known vs Unseen Performance
# %%
plt.figure(figsize=(8, 6))
labels = ['Avg Known F1', 'Unseen F1']
values = [sum(known_f1s)/len(known_f1s) if known_f1s else 0, unseen_f1]
sns.barplot(x=labels, y=values)
plt.title("Known vs Unseen Attack Performance (Random Forest)")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/figures/known_vs_unseen_performance.png")
plt.close()

# %% [markdown]
# ### Conclusion
# This experiment demonstrates how a conventional supervised NIDS (Random Forest) performs when an attack class is completely unseen during training. It typically misclassifies the unseen attack as one of the known classes, resulting in an F1 score of 0 for the unseen attack. This highlights the need for anomaly detection and continual learning.
