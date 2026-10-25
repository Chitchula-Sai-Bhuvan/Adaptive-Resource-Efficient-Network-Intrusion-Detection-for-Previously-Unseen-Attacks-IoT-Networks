# %% [markdown]
# # 02: Preprocessing

# %%
import os
import sys

os.chdir('..')
# Add the src directory to the path so we can import our preprocessing module
sys.path.append(os.path.abspath("."))

from src.preprocessing import Preprocessor

# %% [markdown]
# ### Run the Preprocessing Pipeline
# The `Preprocessor` class handles loading the dataset, cleaning it, creating the known/unseen split, encoding labels, scaling features, and saving the processed output for modeling.

# %%
preprocessor = Preprocessor(config_path="configs/config.yaml")

# Run pipeline
X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = preprocessor.run_pipeline()

# %% [markdown]
# ### Verify Output Shapes
# %%
print(f"X_train shape: {X_train_scaled.shape}")
print(f"y_train shape: {y_train_encoded.shape}")
print(f"X_test shape: {X_test_scaled.shape}")
print(f"y_test shape: {y_test_encoded.shape}")

# %% [markdown]
# Notice that the unseen attack is not present in `y_train_encoded` (we can verify by checking unique values if needed), but is present in `y_test_encoded`.
