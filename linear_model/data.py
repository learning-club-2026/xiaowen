import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from setting import TEST_SIZE, RANDOM_STATE

def load_raw_data(filepath=None):
    if filepath is None:
        current_dir = os.path.dirname(__file__)
        filepath = os.path.join(
            current_dir,
            "data",
            "house-prices-advanced-regression-techniques",
            "train.csv"
        )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在：{filepath}\n请检查目录结构或文件名是否正确。")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    X = df[numeric_cols].values
    y = df['SalePrice'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def load_and_preprocess(filepath=None):
    df = load_raw_data(filepath)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
    }