# data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from setting import *

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    
    # 特征与目标分离
    X = df.drop('target', axis=1)
    y = df['target']
    
    # ---------- 1. 特征分类 ----------
    # 数值特征
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # 类别特征
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # ---------- 2. 构建预处理管道 ----------
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # ---------- 3. 划分训练集和测试集（分层抽样） ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # ---------- 4. 拟合预处理并转换数据 ----------
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 获取最终特征名称（便于权重解释）
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)
    
    print(f"原始特征数: {X.shape[1]} -> 编码后特征数: {X_train_processed.shape[1]}")
    print(f"训练集样本数: {X_train_processed.shape[0]}, 测试集样本数: {X_test_processed.shape[0]}")
    print(f"类别分布（训练集）: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
    print(f"类别分布（测试集）: 0={sum(y_test==0)}, 1={sum(y_test==1)}")
    
    return X_train_processed, X_test_processed, y_train, y_test, feature_names