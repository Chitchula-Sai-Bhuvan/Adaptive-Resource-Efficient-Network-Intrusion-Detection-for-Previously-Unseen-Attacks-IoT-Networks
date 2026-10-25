import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import yaml
import os

class Preprocessor:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def load_and_clean_data(self):
        print(f"Loading data from {self.config['dataset_path']}...")
        df = pd.read_csv(self.config['dataset_path'])
        
        # Replace common placeholders with NaN
        df = df.replace('-', pd.NA)
        
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
            
        # Drop raw IPs to avoid memorization
        if 'src_ip' in df.columns:
            df = df.drop(columns=['src_ip'])
        if 'dst_ip' in df.columns:
            df = df.drop(columns=['dst_ip'])
            
        original_rows = len(df)
        df = df.drop_duplicates()
        print(f"Original rows: {original_rows}, After deduplication: {len(df)}")
        
        return df

    def create_split(self, df):
        target_col = self.config['target_column']
        unseen_attack = self.config['unseen_attack']
        
        # Isolate unseen class
        unseen_df = df[df[target_col] == unseen_attack]
        known_df = df[df[target_col] != unseen_attack]
        
        X_known = known_df.drop(columns=[target_col])
        y_known = known_df[target_col]
        
        X_unseen = unseen_df.drop(columns=[target_col])
        y_unseen = unseen_df[target_col]
        
        X_train, X_test_known, y_train, y_test_known = train_test_split(
            X_known, y_known, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y_known
        )
        
        return X_train, X_test_known, y_train, y_test_known, X_unseen, y_unseen
        
    def fit_transform_features(self, X_train, X_test_known, X_unseen):
        # 1. Drop columns with >50% missing values BASED ON TRAINING SET
        threshold = len(X_train) * 0.5
        valid_cols = X_train.columns[X_train.isna().sum() < threshold]
        
        X_train = X_train[valid_cols].copy()
        X_test_known = X_test_known[valid_cols].copy()
        X_unseen = X_unseen[valid_cols].copy()
        
        # 2. Fill remaining missing values with training medians (for numerics) or modes (for categoricals)
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
        
        for col in num_cols:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test_known[col] = X_test_known[col].fillna(median_val)
            X_unseen[col] = X_unseen[col].fillna(median_val)
            
        for col in cat_cols:
            mode_val = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode_val)
            X_test_known[col] = X_test_known[col].fillna(mode_val)
            X_unseen[col] = X_unseen[col].fillna(mode_val)
            
        # 3. Encode categoricals (like proto, conn_state)
        if len(cat_cols) > 0:
            self.ohe.fit(X_train[cat_cols])
            
            # Transform and create DataFrames
            def encode_df(data_df):
                encoded_array = self.ohe.transform(data_df[cat_cols])
                encoded_cols = self.ohe.get_feature_names_out(cat_cols)
                encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=data_df.index)
                return pd.concat([data_df.drop(columns=cat_cols), encoded_df], axis=1)
                
            X_train = encode_df(X_train)
            X_test_known = encode_df(X_test_known)
            X_unseen = encode_df(X_unseen)
            
        return X_train, X_test_known, X_unseen
        
    def check_overlap(self, X_train, X_test):
        train_hashes = pd.util.hash_pandas_object(X_train, index=False)
        test_hashes = pd.util.hash_pandas_object(X_test, index=False)
        overlap = len(set(train_hashes).intersection(set(test_hashes)))
        overlap_pct = overlap / len(test_hashes) * 100
        print(f"Feature Vector Overlap between Train and Known-Test: {overlap} rows ({overlap_pct:.2f}%)")

    def run_pipeline(self):
        df = self.load_and_clean_data()
        
        X_train, X_test_known, y_train, y_test_known, X_unseen, y_unseen = self.create_split(df)
        
        # Fit Preprocessing strictly on X_train
        X_train, X_test_known, X_unseen = self.fit_transform_features(X_train, X_test_known, X_unseen)
        
        self.check_overlap(X_train, X_test_known)
        
        # Encode Target based ONLY on known classes in training
        # We explicitly ensure unseen_attack is NOT in the encoder's classes
        self.label_encoder.fit(y_train)
        
        unseen_attack = self.config['unseen_attack']
        assert unseen_attack not in self.label_encoder.classes_, f"Error: {unseen_attack} found in trained classes!"
        
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_known_encoded = self.label_encoder.transform(y_test_known)
        
        # We cannot strictly label_encode the unseen test samples because the label encoder has never seen them.
        # But we need a numerical placeholder in y_test for evaluation grouping.
        # We will assign them a special unseen ID (e.g., max_known_id + 1)
        unseen_id = len(self.label_encoder.classes_)
        y_unseen_encoded = np.full(len(y_unseen), unseen_id)
        
        # Re-map the unseen class into the mapping dict manually for the evaluators
        mapping = {str(k): int(v) for k, v in zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))}
        mapping[unseen_attack] = unseen_id
        
        X_test = pd.concat([X_test_known, X_unseen])
        y_test_encoded = np.concatenate([y_test_known_encoded, y_unseen_encoded])
        
        # Save preprocessed data
        os.makedirs("data/processed", exist_ok=True)
        
        X_train.to_csv("data/processed/X_train_unscaled.csv", index=False)
        X_test.to_csv("data/processed/X_test_unscaled.csv", index=False)
        pd.DataFrame(y_train_encoded, columns=[self.config['target_column']]).to_csv("data/processed/y_train.csv", index=False)
        pd.DataFrame(y_test_encoded, columns=[self.config['target_column']]).to_csv("data/processed/y_test.csv", index=False)
        
        with open("data/processed/label_mapping.yaml", "w") as f:
            yaml.dump(mapping, f)
            
        print("Preprocessing complete!")
        return X_train, X_test, y_train_encoded, y_test_encoded

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run_pipeline()
