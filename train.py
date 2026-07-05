import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
import pandas as pd
from setting import *
from data import load_and_preprocess
from model import create_model

# 1. 数据加载
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

# 2. 训练
model = create_model()
model.fit(X_train, y_train)

# 3. 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 4. 评估指标
print("\n评估结果：")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率: {precision_score(y_test, y_pred):.4f}")
print(f"召回率: {recall_score(y_test, y_pred):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred):.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=['无病', '有病']))

# 5. ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], '--', label='随机猜测')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC 曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROC_PLOT, dpi=150)
plt.show()
print(f"\nROC 已保存，AUC = {roc_auc:.3f}")

# 6. 特征权重
coef = pd.DataFrame({'特征': feature_names, '权重': model.weights})
coef = coef.reindex(coef['权重'].abs().sort_values(ascending=False).index)
print("\n特征权重排序：")
print(coef.to_string(index=False))
coef.to_csv("feature_importance.csv", index=False)