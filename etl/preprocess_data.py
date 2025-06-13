import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
import os

def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

def preprocess(config):
    path = 'data/raw_data.csv'
    out_path = config['preprocessed_data_path']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.read_csv(path)

    # Очистка: удаление ненужных колонок
    if 'Id' in df.columns:
        df.drop(columns=['Id'], inplace=True)

    # Кодировка меток
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Деление признаков и меток
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed['diagnosis'] = y.values

    df_processed.to_csv(out_path, index=False)
    print(f"Preprocessed data saved to {out_path}")

if __name__ == "__main__":
    config = load_config()
    preprocess(config)
