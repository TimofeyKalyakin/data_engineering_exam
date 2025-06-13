import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import yaml
import os

def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

def train(config):
    df = pd.read_csv(config['preprocessed_data_path'])
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state'])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)
    joblib.dump(model, config['model_path'])

    print(f"Model saved to {config['model_path']}")

if __name__ == "__main__":
    config = load_config()
    train(config)
