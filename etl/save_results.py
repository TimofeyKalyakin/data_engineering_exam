import yaml
import shutil
import os
from datetime import datetime

def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

def save_results(config):
    model_src = config['model_path']
    metrics_src = config['metrics_path']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest_dir = f"results/run_{timestamp}"
    os.makedirs(dest_dir, exist_ok=True)

    shutil.copy(model_src, os.path.join(dest_dir, 'model.pkl'))
    shutil.copy(metrics_src, os.path.join(dest_dir, 'metrics.json'))

    print(f"Artifacts saved in {dest_dir}")

if __name__ == "__main__":
    config = load_config()
    save_results(config)
