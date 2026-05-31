
import pandas as pd
import numpy as np

#pandas 读取 CSV 文件
data = pd.read_csv("Multivariate_Linear_Regression.csv")

# 拆分变量
X = data[["X1", "X2", "X3", "X4"]].values
y = data["y"].values.reshape(-1, 1)
# 添加截距项
X_b = np.c_[np.ones(len(X)), X] 
# 计算回归系数
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

ntercept = theta[0][0]
coefs = theta[1:].flatten()

print("截距 =", intercept)
print("X1系数 =", coefs[0])
print("X2系数 =", coefs[1])
print("X3系数 =", coefs[2])
print("X4系数 =", coefs[3])
