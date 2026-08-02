import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示异常

# ---------- 1. 从其他模块导入必要的组件 ----------
from setting import HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, DROPOUT_RATE
from setting import LEARNING_RATE, EPOCHS, DEVICE
from data import train_loader, X_test_t, y_test_t, INPUT_DIM
from model import FCNN

print("=" * 50)
print("开始构建训练流水线")
print("=" * 50)

# ---------- 2. 实例化模型 ----------
model = FCNN(
    input_dim=INPUT_DIM,
    hidden1=HIDDEN_DIM1,
    hidden2=HIDDEN_DIM2,
    output_dim=OUTPUT_DIM,
    dropout_rate=DROPOUT_RATE
)
model = model.to(DEVICE)
print("模型结构已初始化:\n", model)

# ---------- 3. 定义损失函数和优化器 ----------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------- 4. 训练循环 ----------
train_losses = []
test_accs = []

print("\n开始训练...")
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # 测试阶段
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t.to(DEVICE))
        test_preds_class = (test_preds >= 0.5).float()
        acc = accuracy_score(y_test_t.numpy(), test_preds_class.cpu().numpy())
        test_accs.append(acc)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | 损失: {epoch_loss:.4f} | 测试准确率: {acc:.4f}")

print("训练完成！")

# ---------- 5. 最终评估 ----------
print("\n" + "=" * 50)
print("最终评估结果")
print("=" * 50)

model.eval()
with torch.no_grad():
    final_preds = model(X_test_t.to(DEVICE))
    final_class = (final_preds >= 0.5).float().cpu().numpy()
    y_test_np = y_test_t.numpy()
    
    final_acc = accuracy_score(y_test_np, final_class)
    cm = confusion_matrix(y_test_np, final_class)
    cr = classification_report(y_test_np, final_class, 
                               target_names=['无心脏病 (0)', '有心脏病 (1)'])
    
    print(f"最终测试准确率: {final_acc:.4f}")
    print("\n混淆矩阵:")
    print(cm)
    print("\n详细分类报告:")
    print(cr)

# ---------- 6. 绘制训练曲线（已修复中文显示） ----------
plt.figure(figsize=(12, 4))

# 左图：训练损失
plt.subplot(1, 2, 1)
plt.plot(train_losses, color='blue', linewidth=2)
plt.xlabel('迭代轮数')        
plt.ylabel('损失值')          
plt.title('训练损失变化趋势')  
plt.grid(True, linestyle='--', alpha=0.6)

# 右图：测试准确率
plt.subplot(1, 2, 2)
plt.plot(test_accs, color='orange', linewidth=2)
plt.xlabel('迭代轮数')        
plt.ylabel('准确率')          
plt.title('测试准确率变化趋势') 
plt.ylim(0.6, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()