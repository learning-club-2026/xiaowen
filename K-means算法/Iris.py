import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. 加载数据
iris = load_iris()
X = iris.data          # 特征矩阵 (150, 4)
y_true = iris.target   
feature_names = iris.feature_names

print("特征名称：", feature_names)
print("前5个样本：\n", X[:5])

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 肘部法则确定最佳 K 值
sse = []  # 簇内误差平方和
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)   #SSE

# 画肘部图
plt.figure(figsize=(6, 4))
plt.plot(K_range, sse, marker='o')
plt.xlabel('聚类数 K')
plt.ylabel('SSE (簇内误差平方和)')
plt.title('肘部法则确定最佳 K')
plt.xticks(K_range)
plt.grid(True)
plt.show()

# 从图上看，拐点在 K=3 附近（Iris 本身也是 3 类）
best_k = 3

# 4. 用 K=3 进行聚类
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_scaled)  # 得到簇标签（0,1,2）

# 5. 查看聚类中心（在标准化后的尺度上）
centers_scaled = kmeans.cluster_centers_
print("聚类中心（标准化后）：\n", centers_scaled)

# 如果想看原始尺度的中心，可以 inverse_transform
centers_original = scaler.inverse_transform(centers_scaled)
print("聚类中心（原始尺度）：\n", centers_original)

# 6. 可视化（用 PCA 降维到 2D 以便绘图）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)  # 对标准化数据做 PCA

plt.figure(figsize=(8, 6))
# 绘制聚类结果
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.7)
# 绘制聚类中心（在 PCA 空间中）
centers_pca = pca.transform(centers_scaled)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, label='聚类中心')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.title(f'K-means 聚类结果 (K={best_k})')
plt.legend()
plt.colorbar(scatter, label='簇标签')
plt.show()

# 7. 与真实标签对比（调整簇标签顺序以匹配真实类别）
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'],
            yticklabels=['True Setosa', 'True Versicolour', 'True Virginica'])
plt.xlabel('预测簇')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.show()

# 计算调整兰德指数 (ARI)，范围 [-1, 1]，越接近 1 表示聚类与真实类别越一致
ari = adjusted_rand_score(y_true, y_pred)
print(f"调整兰德指数 (ARI) = {ari:.3f}")