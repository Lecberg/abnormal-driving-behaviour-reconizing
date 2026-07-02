import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

os.makedirs("0422_gps/result_cnn_lstm_attention", exist_ok=True)

# 全局参数设置
WINDOW_SIZE = 5  # 滑动窗口大小，表示每次处理的连续时间步数
BATCH_SIZE = 64   # 每个批次的样本数
EPOCHS = 30       # 训练轮数
LEARNING_RATE = 0.001  # 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pip设置计算设备（GPU/CPU）
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值，即在多少个epoch验证集性能没有提升就停止训练
EARLY_STOPPING_DELTA = 0.001  # 最小提升阈值，性能提升需要超过此值才算作提升

# 定义输入特征列和标签列
FEATURE_COLS = [
    "海拔", "vss速度", "空档信号", "喇叭信号", "倒挡信号",
    "制动信号", "左转向灯信号", "右转向灯信号", "远光灯信号",
    "近光灯信号", "ACC状态", "与正北方向夹角"
]
LABEL_COLS = ["gps速度", "加速度", "曲折度"]


def preprocess_data(df):
    """
    数据预处理函数
    Args:
        df: 输入的DataFrame数据
    Returns:
        处理后的DataFrame
    """
    # 使用0填充特征和标签中的缺失值
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df[LABEL_COLS] = df[LABEL_COLS].fillna(0)

    # 将ACC状态转换为数值型（1表示开启，0表示关闭）
    df["ACC状态"] = df["ACC状态"].apply(lambda x: 1 if x == "ACC开" else 0)

    return df


def create_label(window_data):
    """
    根据规则创建标签
    Args:
        window_data: 窗口数据
    Returns:
        int: 标签值（0-3）
        0: 正常驾驶
        1: 高速行驶
        2: 急加速/急减速
        3: 曲折行驶
    """
    gps_speed = window_data["gps速度"].values
    acceleration = window_data["加速度"].values
    tortuosity = window_data["曲折度"].values

    # 按优先级顺序判断驾驶状态
    if np.any(tortuosity > 1.5):
        return 3  # 曲折行驶

    if np.any(acceleration > 1.38) or np.any(acceleration < -1.54):
        return 2  # 急加速/急减速

    if np.any(gps_speed >= 80):
        return 1  # 高速行驶

    return 0  # 正常驾驶


class GPSDataset(Dataset):
    """
    自定义GPS数据集类
    用于处理和加载GPS数据
    """
    def __init__(self, data, scaler=None, window_size=WINDOW_SIZE, is_train=False):
        """
        初始化数据集
        Args:
            data: 输入的DataFrame数据
            scaler: 特征标准化器，如果为None且is_train=True则创建新的
            window_size: 滑动窗口大小
            is_train: 是否为训练集
        """
        self.data = data
        self.window_size = window_size
        self.vid_md5_list = data["vid_md5"].unique()
        
        # 处理标准化器
        if scaler is None and is_train:
            self.scaler = StandardScaler()
            self.scaler.fit(data[FEATURE_COLS])
        else:
            self.scaler = scaler
            
        self.samples = self._create_samples()

    def _create_samples(self):
        """
        创建训练样本
        Returns:
            list: 包含(车辆ID, 起始索引, 标签)的样本列表
        """
        samples = []
        for vid in self.vid_md5_list:
            vehicle_data = self.data[self.data["vid_md5"] == vid]
            length = len(vehicle_data)
            if length < self.window_size:
                continue

            # 使用滑动窗口创建样本
            for i in range(length - self.window_size + 1):
                window_data = vehicle_data.iloc[i:i + self.window_size]
                label = create_label(window_data)
                samples.append((vid, i, label))

        return samples

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            tuple: (特征张量, 标签张量)
        """
        vid, start_idx, label = self.samples[idx]
        vehicle_data = self.data[self.data["vid_md5"] == vid]
        window_data = vehicle_data.iloc[start_idx:start_idx + self.window_size]

        # 提取特征并进行标准化
        features = window_data[FEATURE_COLS].values
        features_df = pd.DataFrame(features, columns=FEATURE_COLS)
        if self.scaler is not None:
            features = self.scaler.transform(features_df)
        else:
            features = features_df.values

        # 转换为PyTorch张量
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])

        return features, label


# 在模型类定义前添加注意力类
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attn_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights.squeeze(-1), dim=1)  # (batch_size, seq_len)
        context = torch.bmm(lstm_output.transpose(1, 2), attn_weights.unsqueeze(-1)).squeeze(-1)
        return context, attn_weights


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM混合模型
    结合CNN和LSTM的优势进行序列数据分类
    """
    def __init__(self, input_size, num_classes):
        """
        初始化模型
        Args:
            input_size: 输入特征维度
            num_classes: 分类类别数
        """
        super(CNNLSTMModel, self).__init__()

        # CNN层：用于提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),  # 第一层卷积
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 第二层卷积
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # LSTM层：用于捕获时序依赖
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,  # 简化层数以更好配合注意力
            batch_first=True,
            bidirectional=True
        )

        # 添加注意力层
        self.attention = Attention(hidden_size=256)  # 双向LSTM hidden_size*2

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 保持与注意力输出一致
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据，形状为(batch_size, window_size, input_size)
        Returns:
            tensor: 模型输出
        """
        # 调整维度顺序以适应CNN
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # 调整维度顺序以适应LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 应用注意力机制
        context, _ = self.attention(lstm_out)  # context shape: (batch, 256)
        
        # 通过分类器得到最终输出
        output = self.classifier(context)
        
        return output


def train(model, dataloader, criterion, optimizer, device):
    """
    训练函数
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
    Returns:
        tuple: (epoch损失, epoch准确率)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.squeeze().to(device)

        # 前向传播和反向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    绘制训练曲线
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Final Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Final Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('0422_gps/result_cnn_lstm_attention/final_training_curves.png')
    plt.close()


def evaluate(model, dataloader, criterion, device):
    """
    评估函数
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
    Returns:
        tuple: (评估损失, 评估准确率)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 计算统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集预测结果用于绘制混淆矩阵
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc


def plot_confusion_matrix(true_labels, pred_labels):
    """
    绘制混淆矩阵
    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
    """
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'High Speed', 'Acceleration', 'Tortuosity'],
                yticklabels=['Normal', 'High Speed', 'Acceleration', 'Tortuosity'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('0422_gps/result_cnn_lstm_attention/confusion_matrix.png')
    plt.close()


class EarlyStopping:
    """
    早停机制类
    用于监控验证集性能，在性能不再提升时提前结束训练
    """
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, delta=EARLY_STOPPING_DELTA):
        """
        初始化早停机制
        Args:
            patience: 容忍验证集性能不提升的轮数
            delta: 性能提升的最小阈值
        """
        self.patience = patience
        self.delta = delta
        self.best_val_acc = None
        self.counter = 0
        self.best_model = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        """
        检查是否应该早停
        Args:
            val_acc: 当前epoch的验证集准确率
            model: 当前模型
        Returns:
            bool: 是否应该停止训练
        """
        if self.best_val_acc is None:
            self.best_val_acc = val_acc
            self.save_checkpoint(model)
            return False

        if val_acc > self.best_val_acc + self.delta:
            self.best_val_acc = val_acc
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def save_checkpoint(self, model):
        """
        保存最佳模型
        Args:
            model: 当前模型
        """
        torch.save(model.state_dict(), "0422_gps/result_cnn_lstm_attention/best_model.pth")
        print("保存最佳模型")


def main():
    """
    主函数：执行完整的训练和评估流程
    """
    # 加载和预处理数据
    df = pd.read_csv("0422_gps/data/gps_features.csv")
    df = preprocess_data(df)

    # 按车辆ID划分数据集
    unique_vids = df["vid_md5"].unique()
    train_vids, test_vids = train_test_split(unique_vids, test_size=0.2, random_state=42)
    val_vids, test_vids = train_test_split(test_vids, test_size=0.5, random_state=42)

    # 根据车辆ID分割数据
    train_df = df[df["vid_md5"].isin(train_vids)]
    val_df = df[df["vid_md5"].isin(val_vids)]
    test_df = df[df["vid_md5"].isin(test_vids)]

    # 创建训练集的标准化器
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])

    # 创建数据集和数据加载器
    train_dataset = GPSDataset(train_df, scaler=scaler, is_train=True)
    val_dataset = GPSDataset(val_df, scaler=scaler, is_train=False)
    test_dataset = GPSDataset(test_df, scaler=scaler, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数和优化器
    input_size = len(FEATURE_COLS)
    num_classes = 4  # 0: 正常, 1: 高速, 2: 急加速/急减速, 3: 曲折行驶
    model = CNNLSTMModel(input_size, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 初始化早停机制
    early_stopping = EarlyStopping()

    # 训练循环
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # 训练阶段
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 检查是否需要早停
        if early_stopping(val_acc, model):
            print("触发早停机制！")
            break

    # 绘制训练过程曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 在测试集上评估最佳模型
    model.load_state_dict(torch.load("0422_gps/result_cnn_lstm_attention/best_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
