# 训练过程
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_and_preprocess
from model import LinearRegression
from setting import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    FIGURE_SAVE_DIR
)


def train() -> None:
    # 1. 加载数据
    data = load_and_preprocess()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    X_final_test = data["X_final_test"]   # 用于最终提交

    # 2. 初始化模型
    model = LinearRegression(
        n_features=X_train.shape[1],
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED,
    )

    # 3. 初始化记录容器
    epochs = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # 4. 确保输出目录存在
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    log_path = os.path.join(FIGURE_SAVE_DIR, "train_report.txt")

    # 5. 写入日志表头
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"本次 RANDOM_SEED: {RANDOM_SEED}\n")
        f.write(f"{'Epoch':>6} | {'Train Loss':>12} | {'Test Loss':>12} | {'Train Acc':>10} | {'Test Acc':>10}\n")
        f.write("-" * 62 + "\n")

    # 6. 打印控制台表头
    print(f"本次 RANDOM_SEED: {RANDOM_SEED}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Test Loss':>12} | {'Train Acc':>10} | {'Test Acc':>10}")
    print("-" * 62)

    # 7. 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.fit_epoch(X_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_SEED + epoch)

        train_loss, train_acc = model.evaluate(X_train, y_train)
        test_loss, test_acc = model.evaluate(X_test, y_test)

        epochs.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 写入日志
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch:>6} | {train_loss:>12.4f} | {test_loss:>12.4f} | "
                f"{train_acc:>9.2f}% | {test_acc:>9.2f}%\n"
            )

        # 控制台输出（每10个epoch打印一次，避免刷屏）
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{epoch:>6} | {train_loss:>12.4f} | {test_loss:>12.4f} | "
                f"{train_acc:>9.2f}% | {test_acc:>9.2f}%"
            )

    # 8. 最终评估
    y_test_pred = model.predict(X_test)
    rmse = float(np.sqrt(model.mse(y_test, y_test_pred)))
    r2 = model.r2_score(y_test, y_test_pred)

    # 9. 打印最终结果
    print()
    print("=== 线性回归训练结果 (梯度下降) ===")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"训练样本: {len(y_train)}, 测试样本: {len(y_test)}")
    print(f"迭代次数: {EPOCHS}, 学习率: {LEARNING_RATE}, batch_size: {BATCH_SIZE}")
    print(f"最终训练 Loss: {train_loss:.4f}")
    print(f"最终测试 Loss: {test_loss:.4f}")
    print(f"最终训练准确率: {train_acc:.2f}%")
    print(f"最终测试准确率: {test_acc:.2f}%")
    print(f"截距 b: {model.b:.4f}")
    print(f"权重 w (标准化特征空间): {np.array2string(model.w, precision=4, separator=' ')}")
    print()
    print(f"测试集 MSE:  {test_loss:.4f}")
    print(f"测试集 RMSE: {rmse:.4f}")
    print(f"测试集 R2:   {r2:.4f}")

    # 10. 写入最终结果到日志
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n=== 线性回归训练结果 (梯度下降) ===\n")
        f.write(f"RANDOM_SEED: {RANDOM_SEED}\n")
        f.write(f"训练样本: {len(y_train)}, 测试样本: {len(y_test)}\n")
        f.write(f"迭代次数: {EPOCHS}, 学习率: {LEARNING_RATE}, batch_size: {BATCH_SIZE}\n")
        f.write(f"最终训练 Loss: {train_loss:.4f}\n")
        f.write(f"最终测试 Loss: {test_loss:.4f}\n")
        f.write(f"最终训练准确率: {train_acc:.2f}%\n")
        f.write(f"最终测试准确率: {test_acc:.2f}%\n")
        f.write(f"截距 b: {model.b:.4f}\n")
        f.write(f"权重 w (标准化特征空间): {np.array2string(model.w, precision=4, separator=' ')}\n")
        f.write(f"测试集 MSE:  {test_loss:.4f}\n")
        f.write(f"测试集 RMSE: {rmse:.4f}\n")
        f.write(f"测试集 R2:   {r2:.4f}\n")

    # 11. 绘制 Accuracy 曲线
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_accs, label="Train", color="#1f77b4", linewidth=2)
    plt.plot(epochs, test_accs, label="Test", color="#d62728", linewidth=2)
    plt.title("Train vs Test R2 Score (%)", fontsize=14, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("R2 Accuracy (%)", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_SAVE_DIR, "accuracy.png"), dpi=150)
    plt.close()

    # 12. 绘制 Loss 曲线（对数尺度）
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, test_losses, label="Test Loss", color="#d62728", linewidth=2)
    plt.yscale('log')
    plt.title("Train vs Test Loss (Log Scale)", fontsize=14, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss (log)", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_SAVE_DIR, "loss.png"), dpi=150)
    plt.close()


train()