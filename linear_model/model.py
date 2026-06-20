# 模型构建
from __future__ import annotations

from typing import Optional

import numpy as np


class LinearRegression:
    """多元线性回归，使用 mini-batch 梯度下降训练。"""

    def __init__(self, n_features: int, learning_rate: float, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(scale=0.01, size=n_features)
        self.b = 0.0
        self.learning_rate = learning_rate

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        y_pred = self.predict(X)
        loss = self.mse(y, y_pred)
        accuracy = self.r2_score(y, y_pred) * 100
        return loss, accuracy

    def fit_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> None:
        n = len(y_batch)
        y_pred = self.predict(X_batch)
        error = y_pred - y_batch

        grad_w = (2.0 / n) * (X_batch.T @ error)
        grad_b = (2.0 / n) * np.sum(error)

        self.w -= self.learning_rate * grad_w
        self.b -= self.learning_rate * grad_b

    def fit_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        seed: Optional[int] = None,
    ) -> None:
        n_samples = len(y)
        indices = np.arange(n_samples)
        rng = np.random.default_rng(seed)

        rng.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            self.fit_batch(X[batch_idx], y[batch_idx])