import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class NPYDepressionDataset(Dataset):
    """直接从npy文件加载的抑郁症分类数据集（修复版）"""

    def __init__(self, features_base_path, split='train'):
        """
        初始化数据集

        Args:
            features_base_path: 特征文件根目录
            split: 数据集分割 ('train' 或 'val')
        """
        self.split = split
        self.split_path = os.path.join(features_base_path, split)

        if not os.path.exists(self.split_path):
            raise ValueError(f"❌ 数据集路径不存在: {self.split_path}")

        # 加载抑郁和非抑郁特征文件
        depressed_path = os.path.join(self.split_path, 'depressed_embeddings.npy')
        non_depressed_path = os.path.join(self.split_path, 'non_depressed_embeddings.npy')

        if not os.path.exists(depressed_path):
            raise ValueError(f"❌ 抑郁特征文件不存在: {depressed_path}")
        if not os.path.exists(non_depressed_path):
            raise ValueError(f"❌ 非抑郁特征文件不存在: {non_depressed_path}")

        # 加载特征数据
        depressed_features = np.load(depressed_path)
        non_depressed_features = np.load(non_depressed_path)

        print(f"🔍 原始特征形状 - 抑郁: {depressed_features.shape}, 非抑郁: {non_depressed_features.shape}")

        # 修复：确保特征是二维的 (n_samples, feature_dim)
        if len(depressed_features.shape) == 1:
            depressed_features = depressed_features.reshape(-1, 1)
        if len(non_depressed_features.shape) == 1:
            non_depressed_features = non_depressed_features.reshape(-1, 1)

        # 合并特征和创建标签
        self.features = np.concatenate([depressed_features, non_depressed_features], axis=0)
        self.labels = np.concatenate([
            np.ones(len(depressed_features)),  # 抑郁标签为1
            np.zeros(len(non_depressed_features))  # 非抑郁标签为0
        ], axis=0)

        print(f"✅ 成功加载 {split} 数据集:")
        print(f"   - 抑郁样本数: {len(depressed_features)}")
        print(f"   - 非抑郁样本数: {len(non_depressed_features)}")
        print(f"   - 总样本数: {len(self.features)}")
        print(f"   - 特征维度: {self.features.shape[1]}")
        print(f"   - 特征数组形状: {self.features.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # 修复：确保特征是1D数组（即使维度为1）
        if len(feature.shape) > 0:
            feature = feature.flatten()

        # 转换为PyTorch张量
        feature_tensor = torch.FloatTensor(feature)
        label_tensor = torch.LongTensor([int(label)])  # 确保标签是整数

        return feature_tensor, label_tensor.squeeze()


class DepressionClassifier(nn.Module):
    """抑郁症二分类模型（适配低维特征）"""

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        """
        初始化分类器（适配低维特征）

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表（根据输入维度调整）
            dropout_rate: dropout比率
        """
        super(DepressionClassifier, self).__init__()

        # 根据输入维度动态调整网络结构
        if input_dim <= 10:  # 低维特征
            hidden_dims = [max(32, input_dim * 4), max(16, input_dim * 2)]
        elif input_dim <= 100:  # 中等维度
            hidden_dims = [256, 128]
        else:  # 高维特征
            hidden_dims = [512, 256, 128]

        layers = []
        prev_dim = input_dim

        # 动态构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if hidden_dim > 1 else nn.Identity(),  # 低维时跳过BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 2))  # 二分类输出

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # 修复：确保输入维度正确
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # 如果是1D，添加batch维度
        return self.classifier(x)


class DepressionTrainer:
    """抑郁症分类训练器（修复版）"""

    def __init__(self, model, train_loader, val_loader, device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self):
        """训练一个epoch（修复版）"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"训练中")
        for batch_idx, (features, labels) in enumerate(pbar):
            try:
                # 修复：确保特征形状正确
                if len(features.shape) == 1:
                    features = features.unsqueeze(1)  # 如果是1D，添加特征维度

                features = features.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 统计信息
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # 更新进度条
                current_acc = 100.0 * correct_predictions / total_samples
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

            except Exception as e:
                print(f"❌ 批处理 {batch_idx} 出错: {e}")
                print(f"   特征形状: {features.shape}")
                print(f"   标签形状: {labels.shape}")
                continue

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_samples

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)

        return epoch_loss, epoch_accuracy

    def validate_epoch(self):
        """验证一个epoch（修复版）"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"验证中")
            for features, labels in pbar:
                try:
                    # 修复：确保特征形状正确
                    if len(features.shape) == 1:
                        features = features.unsqueeze(1)

                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    # 更新进度条
                    current_acc = 100.0 * correct_predictions / total_samples
                    pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})

                except Exception as e:
                    print(f"❌ 验证批处理出错: {e}")
                    continue

        epoch_loss = running_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        epoch_accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0

        # 计算其他指标
        precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0) if len(
            all_labels) > 0 else 0
        recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0) if len(
            all_labels) > 0 else 0
        f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0) if len(all_labels) > 0 else 0

        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)

        return epoch_loss, epoch_accuracy, precision, recall, f1, all_predictions, all_labels

    def train(self, epochs=50, save_path='best_depression_classifier.pth'):
        """完整训练流程"""
        print("🚀 开始训练抑郁症分类模型...")
        best_accuracy = 0.0

        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc, precision, recall, f1, _, _ = self.validate_epoch()

            # 学习率调整
            self.scheduler.step(val_loss)

            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")

            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'input_dim': self.model.classifier[0].in_features
                }, save_path)
                print(f"💾 保存最佳模型，准确率: {best_accuracy:.2f}%")

        print(f"\n✅ 训练完成！最佳验证准确率: {best_accuracy:.2f}%")


def debug_dataset_shapes(features_base_path):
    """调试函数：检查数据集形状"""
    print("🐛 调试信息:")

    for split in ['train', 'val']:
        split_path = os.path.join(features_base_path, split)
        if os.path.exists(split_path):
            depressed_path = os.path.join(split_path, 'depressed_embeddings.npy')
            non_depressed_path = os.path.join(split_path, 'non_depressed_embeddings.npy')

            if os.path.exists(depressed_path):
                depressed_features = np.load(depressed_path)
                print(f"{split}/depressed.npy - 形状: {depressed_features.shape}, 数据类型: {depressed_features.dtype}")

            if os.path.exists(non_depressed_path):
                non_depressed_features = np.load(non_depressed_path)
                print(
                    f"{split}/non_depressed.npy - 形状: {non_depressed_features.shape}, 数据类型: {non_depressed_features.dtype}")


def main():
    """主函数（修复版）"""
    # 配置路径
    FEATURES_BASE_PATH = "./2-Dep-Classification/"  # 特征文件根目录
    MODEL_SAVE_PATH = "best_depression_classifier.pth"

    # 调试：检查数据集形状
    debug_dataset_shapes(FEATURES_BASE_PATH)

    # 检查路径是否存在
    if not os.path.exists(FEATURES_BASE_PATH):
        print(f"❌ 特征路径不存在: {FEATURES_BASE_PATH}")
        return

    try:
        # 创建数据集
        train_dataset = NPYDepressionDataset(FEATURES_BASE_PATH, split='train')
        val_dataset = NPYDepressionDataset(FEATURES_BASE_PATH, split='val')

        # 获取特征维度
        feature_dim = train_dataset.features.shape[1]
        print(f"🔢 最终使用的特征维度: {feature_dim}")

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 使用设备: {device}")

        # 创建数据加载器（修复：设置drop_last=True避免最后一个batch问题）
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

        print(f"📊 训练集大小: {len(train_dataset)}")
        print(f"📊 验证集大小: {len(val_dataset)}")
        print(f"📦 批大小: {batch_size}")

        # 创建模型
        model = DepressionClassifier(input_dim=feature_dim)
        print(f"🧠 模型结构:")
        print(model)
        print(f"📐 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 调整学习率（根据特征维度）
        learning_rate = 1e-3 if feature_dim <= 10 else 1e-4
        print(f"📚 使用学习率: {learning_rate}")

        # 创建训练器并开始训练
        trainer = DepressionTrainer(model, train_loader, val_loader, device=device, learning_rate=learning_rate)
        trainer.train(epochs=30, save_path=MODEL_SAVE_PATH)  # 减少epoch数用于测试

    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()