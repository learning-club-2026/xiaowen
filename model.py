# model.py
import numpy as np

class LogisticRegressionCustom:
    def __init__(self, C=1.0, max_iter=1000, learning_rate=0.01, penalty='l2', random_state=None):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Sigmoid 函数
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _loss(self, X, y, w, b):
        # 对数损失 + L2 正则项
        m = X.shape[0]
        p = self._sigmoid(np.dot(X, w) + b)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = (1 / self.C) * 0.5 * np.sum(w ** 2) if self.penalty == 'l2' else 0
        return log_loss + reg

    def fit(self, X, y):
        # 批量梯度下降
        if self.random_state:
            np.random.seed(self.random_state)
        m, n = X.shape
        self.weights = np.random.randn(n) * 0.01
        self.bias = 0.0

        for _ in range(self.max_iter):
            # 前向传播
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)

            # 梯度计算
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            if self.penalty == 'l2':
                dw += (1/self.C) * self.weights

            # 参数更新
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        # 预测概率
        linear = np.dot(X, self.weights) + self.bias
        prob_pos = self._sigmoid(linear)
        return np.c_[1 - prob_pos, prob_pos]

    def predict(self, X):
        # 预测类别
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def create_model():
    from setting import LR_C, LR_MAX_ITER, LR_LEARNING_RATE, LR_PENALTY, RANDOM_SEED
    return LogisticRegressionCustom(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        learning_rate=LR_LEARNING_RATE,
        penalty=LR_PENALTY,
        random_state=RANDOM_SEED
    )