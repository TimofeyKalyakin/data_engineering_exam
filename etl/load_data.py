import pandas as pd
import os
import yaml
import shutil

def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

def load_data(config):
    source_path = config['local_data_path']
    target_path = 'data/raw_data.csv'
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    shutil.copy(source_path, target_path)
    print(f"Copied data to {target_path}")

if __name__ == "__main__":
    config = load_config()
    load_data(config)
