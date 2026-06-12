import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv("Multivariate_Linear_Regression.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# ===================== 超参数 =====================
lr = 0.0001
epochs = 200


# ===================== 梯度下降 =====================
def gradient_descent(X, y, lr, epochs):
    w = np.zeros(4)
    b = 0.0
    n = len(X)
    history = {'w': [], 'b': [], 'loss': []}

    for epoch in range(epochs):
        # 预测
        y_pred = np.dot(X, w) + b

        # 损失
        loss = np.mean((y_pred - y) ** 2)

        # 梯度
        dw = (2 / n) * np.dot(X.T, y_pred - y)
        db = (2 / n) * np.sum(y_pred - y)

        # 更新
        w -= lr * dw
        b -= lr * db

        # 记录
        history['w'].append(w.copy())
        history['b'].append(b)
        history['loss'].append(loss)

        # ===================== 每次迭代输出（6位小数） =====================
        print(
            f"迭代 {epoch + 1:3d} | w1={w[0]:.6f} w2={w[1]:.6f} w3={w[2]:.6f} w4={w[3]:.6f} | b={b:.6f} | loss={loss:.6f}")

    return w, b, history


# 训练
w_final, b_final, history = gradient_descent(X, y, lr, epochs)

# ===================== 最终结果 =====================
print("\n" + "=" * 80)
print("训练完成（200 次迭代）")
print(f"拟合公式：y = {b_final:.6f} + {w_final[0]:.6f}x1 + {w_final[1]:.6f}x2 + {w_final[2]:.6f}x3 + {w_final[3]:.6f}x4")
print("=" * 80)