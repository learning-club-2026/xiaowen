import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from setting import BATCH_SIZE, RANDOM_SEED, DEVICE

# ---------- 1. 读取数据 ----------
print("=" * 50)
print("正在加载数据...")

script_dir = os.path.dirname(os.path.abspath(__file__))
heart_path = os.path.join(script_dir, 'heart.csv')

df = pd.read_csv(heart_path)

# ---------- 2. 使用全部原始特征（除 target 外） ----------
X = df.drop('target', axis=1).values
y = df['target'].values

feature_names = df.drop('target', axis=1).columns.tolist()
print(f"使用的特征: {feature_names}")
print(f"特征数量: {len(feature_names)}")

# ---------- 3. 划分训练集和测试集 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)

# ---------- 4. 数据标准化 ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- 5. 转换为 PyTorch 张量 ----------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ---------- 6. 打包为 DataLoader ----------
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- 7. 导出 ----------
INPUT_DIM = X_train.shape[1]

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print(f"输入特征维度: {INPUT_DIM}")
print("=" * 50)