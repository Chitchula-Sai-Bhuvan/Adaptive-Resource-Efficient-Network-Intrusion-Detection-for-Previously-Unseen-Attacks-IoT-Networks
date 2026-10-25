from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml
import joblib
import pandas as pd
import os

class RFModel:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        rf_config = self.config.get('random_forest', {})
        self.model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', None),
            random_state=self.config.get('random_state', 42),
            n_jobs=-1
        )
        
    def train(self, X_train, y_train):
        print("Training Random Forest baseline...")
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test, label_mapping_path="data/processed/label_mapping.yaml"):
        print("Evaluating Random Forest baseline...")
        y_pred = self.model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Load label mapping to evaluate per-class
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, "r") as f:
                mapping = yaml.safe_load(f)
                reverse_mapping = {v: k for k, v in mapping.items()}
                
            labels = list(range(len(mapping)))
            
            per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0, labels=labels)
            
            # Identify known vs unseen attack performance if possible
            # The unseen attack should have 0 recall since it wasn't in the training set
            unseen_attack = self.config['unseen_attack']
            
            for encoded_label, f1 in enumerate(per_class_f1):
                class_name = reverse_mapping.get(encoded_label, str(encoded_label))
                results[f'f1_{class_name}'] = float(f1)
                
        return results
        
    def save(self, path="models/random_forest_baseline.pkl"):
        print(f"Saving model to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
    def load(self, path="models/random_forest_baseline.pkl"):
        print(f"Loading model from {path}...")
        self.model = joblib.load(path)

if __name__ == "__main__":
    # Test script execution
    X_train = pd.read_csv("data/processed/X_train_unscaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test_unscaled.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    rf = RFModel()
    rf.train(X_train, y_train)
    
    results = rf.evaluate(X_test, y_test)
    print("Results:", results)
    
    rf.save()
    
    # Save results
    import json
    with open("results/metrics/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
