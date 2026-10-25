# %% [markdown]
# # 01: Exploratory Data Analysis (EDA) on Real Dataset

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

os.chdir('..')

# %%
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
dataset_path = config['dataset_path']
print(f"Loading actual dataset from {dataset_path}...")
df = pd.read_csv(dataset_path)

# %% [markdown]
# ### Dataset Characteristics

# %%
num_rows, num_cols = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")

print("\nFeature Names:")
print(df.columns.tolist())

# %% [markdown]
# ### Data Types and Missing Values
# The dataset may contain '-' instead of NaNs, which we will replace before checking for nulls.

# %%
# Replace common string placeholders with proper NaN for analysis
df = df.replace('-', pd.NA)

missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

duplicate_rows = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_rows}")

# Save basic summary
os.makedirs("results/tables", exist_ok=True)
summary_df = pd.DataFrame({
    'Metric': ['Total Rows', 'Total Columns', 'Duplicate Rows'],
    'Value': [num_rows, num_cols, duplicate_rows]
})
summary_df.to_csv("results/tables/dataset_summary.csv", index=False)

# %% [markdown]
# ### Class Distribution
# Exploring the actual target labels present.

# %%
target_col = config['target_column']
print(f"\nTarget column: '{target_col}'")

class_counts = df[target_col].value_counts()
class_percentages = df[target_col].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    'Attack Class': class_counts.index,
    'Count': class_counts.values,
    'Percentage': class_percentages.values
})

print("\n--- Actual Class Distribution ---")
print(dist_df)

dist_df.to_csv("results/tables/class_distribution.csv", index=False)

# %%
plt.figure(figsize=(12, 6))
sns.barplot(data=dist_df, x='Attack Class', y='Count')
plt.title(f"Class Distribution in {dataset_path}")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/class_distribution.png")
plt.close()

# %% [markdown]
# ### Selection of Unseen Attack
# Based on the actual distribution above, we have verified that 'mitm' exists and has sufficient samples (1043). 
# It is configured in `configs/config.yaml` as the hold-out class for the unseen attack experiment.
