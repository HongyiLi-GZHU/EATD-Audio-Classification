import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 加载嵌入向量
base_dir = "/share/workspace/lixiang/HongyiLi/HongyiLi/ML/Qwen2-Audio-Embedding"
negative_embeddings = np.load(os.path.join(base_dir, "negative_out_embeddings.npy"))
neutral_embeddings = np.load(os.path.join(base_dir, "neutral_out_embeddings.npy"))
positive_embeddings = np.load(os.path.join(base_dir, "positive_out_embeddings.npy"))

# 创建标签
negative_labels = np.zeros(len(negative_embeddings))  # 0 for negative
neutral_labels = np.ones(len(neutral_embeddings))  # 1 for neutral
positive_labels = np.full(len(positive_embeddings), 2)  # 2 for positive

# 合并所有嵌入和标签
all_embeddings = np.concatenate([negative_embeddings, neutral_embeddings, positive_embeddings], axis=0)
all_labels = np.concatenate([negative_labels, neutral_labels, positive_labels], axis=0)

print(f"数据集大小: {all_embeddings.shape}")
print(
    f"标签分布: Negative={np.sum(all_labels == 0)}, Neutral={np.sum(all_labels == 1)}, Positive={np.sum(all_labels == 2)}")

# 转换为PyTorch张量
X = torch.tensor(all_embeddings, dtype=torch.float32)
y = torch.tensor(all_labels, dtype=torch.long)

# 创建数据集
dataset = TensorDataset(X, y)

# 计算每个类别的样本数量
class_counts = [np.sum(all_labels == i) for i in range(3)]
print(f"各类别样本数量: {class_counts}")

# 按类别划分训练集和测试集（每个类别70%训练，30%测试）
train_indices = []
test_indices = []

for class_idx in range(3):
    class_indices = np.where(all_labels == class_idx)[0]
    np.random.shuffle(class_indices)

    # 计算训练样本数量（70%）
    train_count = int(0.7 * len(class_indices))

    train_indices.extend(class_indices[:train_count])
    test_indices.extend(class_indices[train_count:])

# 创建训练集和测试集
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义简单的线性模型
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# 初始化模型
input_dim = all_embeddings.shape[1]
num_classes = 3
model = LinearClassifier(input_dim, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
train_losses = []
train_accuracies = []

print("开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

print("训练完成!")

# 评估模型
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# 计算整体指标
overall_accuracy = accuracy_score(all_targets, all_predictions)
overall_f1 = f1_score(all_targets, all_predictions, average='weighted')

print(f"\n整体评估结果:")
print(f"整体准确率: {overall_accuracy:.4f}")
print(f"加权F1分数: {overall_f1:.4f}")

# 计算每个类别的指标
class_names = ['Negative', 'Neutral', 'Positive']
class_accuracy = []
class_f1 = []

for class_idx in range(3):
    # 创建二分类标签（当前类别 vs 其他）
    binary_targets = np.array(all_targets) == class_idx
    binary_predictions = np.array(all_predictions) == class_idx

    # 计算准确率和F1分数
    acc = accuracy_score(binary_targets, binary_predictions)
    f1 = f1_score(binary_targets, binary_predictions)

    class_accuracy.append(acc)
    class_f1.append(f1)

    print(f"\n{class_names[class_idx]}类别:")
    print(f"  准确率: {acc:.4f}")
    print(f"  F1分数: {f1:.4f}")

# 生成详细的分类报告
print("\n详细分类报告:")
print(classification_report(all_targets, all_predictions, target_names=class_names))

# 绘制训练过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title('训练准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'training_curve.png'))
plt.show()

# 保存模型
model_path = os.path.join(base_dir, 'linear_classifier.pth')
torch.save(model.state_dict(), model_path)
print(f"\n模型已保存至: {model_path}")

# 保存评估结果
results_df = pd.DataFrame({
    'Class': class_names,
    'Accuracy': class_accuracy,
    'F1_Score': class_f1
})

results_df.loc['Overall'] = ['All', overall_accuracy, overall_f1]
results_path = os.path.join(base_dir, 'evaluation_results.csv')
results_df.to_csv(results_path, index=False)
print(f"评估结果已保存至: {results_path}")