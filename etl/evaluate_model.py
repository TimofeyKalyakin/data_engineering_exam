import pandas as pd
import joblib
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import json
import os

def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

def evaluate_model(config):
    df = pd.read_csv(config['preprocessed_data_path'])
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state'])

    model = joblib.load(config['model_path'])
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    os.makedirs(os.path.dirname(config['metrics_path']), exist_ok=True)
    with open(config['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics saved:")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    config = load_config()
    evaluate_model(config)
