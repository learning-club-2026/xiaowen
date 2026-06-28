#线性回归模型
import numpy as np

class LinearRegression:
    def __init__(self, n_features: int, learning_rate: float = 0.01, seed: int = 42):
        self.lr = learning_rate
        np.random.seed(seed)
        self.w = np.random.randn(n_features) * 0.01   # 小随机初始化
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回预测值 y = Xw + b"""
        return X @ self.w + self.b

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """返回 (MSE, R2*100) 用于监控"""
        pred = self.predict(X)
        loss = self.mse(y, pred)
        acc = self.r2_score(y, pred) * 100
        return loss, acc

    def fit_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = None, seed: int = None):
        """执行一个epoch的梯度下降（可小批量）"""
        m = X.shape[0]
        if batch_size is None:
            batch_size = m
        # 打乱数据
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            pred = self.predict(X_batch)
            error = pred - y_batch
            # 梯度
            dw = (X_batch.T @ error) / len(y_batch)
            db = np.mean(error)
            # 更新
            self.w -= self.lr * dw
            self.b -= self.lr * db