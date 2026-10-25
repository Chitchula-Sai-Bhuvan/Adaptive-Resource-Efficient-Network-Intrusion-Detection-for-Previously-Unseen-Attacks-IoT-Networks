# %% [markdown]
# # 03: Random Forest Baseline

# %%
import pandas as pd
import json
import os
import sys

os.chdir('..')
sys.path.append(os.path.abspath("."))
from src.models.random_forest import RFModel

# %% [markdown]
# ### Load Processed Data
# %%
X_train = pd.read_csv("data/processed/X_train_unscaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test = pd.read_csv("data/processed/X_test_unscaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# %% [markdown]
# ### Initialize and Train Random Forest
# %%
rf_model = RFModel(config_path="configs/config.yaml")
rf_model.train(X_train, y_train)

# %% [markdown]
# ### Evaluate on Test Set
# %%
results = rf_model.evaluate(X_test, y_test, label_mapping_path="data/processed/label_mapping.yaml")

print("--- Random Forest Baseline Results ---")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Macro Precision: {results['precision_macro']:.4f}")
print(f"Macro Recall: {results['recall_macro']:.4f}")
print(f"Macro F1: {results['f1_macro']:.4f}")

# %% [markdown]
# ### Save Model and Results
# %%
rf_model.save("models/random_forest_baseline.pkl")

os.makedirs("results/metrics", exist_ok=True)
with open("results/metrics/baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Saved baseline results to results/metrics/baseline_results.json")
