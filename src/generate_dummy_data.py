import pandas as pd
import numpy as np
import os
import yaml

def generate_dummy_data(output_path, num_samples=10000):
    np.random.seed(42)
    
    # Common features in network intrusion datasets
    features = [
        'flow_duration', 'header_length', 'protocol_type', 'duration', 
        'rate', 'srate', 'drate', 'fin_flag_number', 'syn_flag_number', 
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
        'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 
        'fin_count', 'urg_count', 'rst_count', 'http', 'https', 'dns', 
        'telnet', 'smtp', 'ssh', 'irc', 'tcp', 'udp', 'dhcp', 'arp', 
        'icmp', 'ipv', 'llc'
    ]
    
    # Attack classes based on CICIoT2023 dataset types
    classes = [
        'Benign', 'DDoS', 'DoS', 'Brute-force', 'Reconnaissance', 
        'Spoofing', 'Web-based', 'Mirai'
    ]
    
    # Assign probabilities to classes (Benign usually majority)
    probs = [0.4, 0.15, 0.15, 0.05, 0.1, 0.05, 0.05, 0.05]
    
    # Generate labels
    labels = np.random.choice(classes, size=num_samples, p=probs)
    
    # Generate synthetic features
    data = {}
    for feature in features:
        # Add some slight variation depending on the class to make the model learn something
        base_feature = np.random.normal(loc=100, scale=50, size=num_samples)
        
        # Apply some class-specific shifts
        shift = np.zeros(num_samples)
        shift[labels == 'DDoS'] += 50
        shift[labels == 'Mirai'] -= 20
        shift[labels == 'Benign'] -= 30
        
        data[feature] = np.abs(base_feature + shift) # Keep features positive
    
    df = pd.DataFrame(data)
    df['label'] = labels
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {num_samples} samples at {output_path}")

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    generate_dummy_data(config['dataset_path'])
