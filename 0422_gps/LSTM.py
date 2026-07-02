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
import joblib

os.makedirs("0422_gps/result_lstm", exist_ok=True)

# 参数设置
WINDOW_SIZE = 10
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特征列和标签定义
FEATURE_COLS = [
    "海拔", "vss速度", "空档信号", "喇叭信号", "倒挡信号",
    "制动信号", "左转向灯信号", "右转向灯信号", "远光灯信号",
    "近光灯信号", "ACC状态", "与正北方向夹角"
]
LABEL_COLS = ["gps速度", "加速度", "曲折度"]


# 数据预处理
def preprocess_data(df):
    # 填充缺失值
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df[LABEL_COLS] = df[LABEL_COLS].fillna(0)

    # 转换ACC状态
    df["ACC状态"] = df["ACC状态"].apply(lambda x: 1 if x == "ACC开" else 0)

    return df


# 创建标签
def create_label(window_data):
    gps_speed = window_data["gps速度"].values
    acceleration = window_data["加速度"].values
    tortuosity = window_data["曲折度"].values

    # 规则3: 曲折度 > 1.5
    if np.any(tortuosity > 1.5):
        return 3

    # 规则2: 加速度 > 1.38 或 < -1.54
    if np.any(acceleration > 1.38) or np.any(acceleration < -1.54):
        return 2

    # 规则1: gps速度 >= 80
    if np.any(gps_speed >= 80):
        return 1

    # 正常情况
    return 0


class GPSDataset(Dataset):
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
        samples = []
        for vid in self.vid_md5_list:
            vehicle_data = self.data[self.data["vid_md5"] == vid]
            length = len(vehicle_data)

            if length < self.window_size:
                continue
            for i in range(length - self.window_size + 1):
                window_data = vehicle_data.iloc[i:i + self.window_size]
                label = create_label(window_data)
                samples.append((vid, i, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, start_idx, label = self.samples[idx]
        vehicle_data = self.data[self.data["vid_md5"] == vid]
        window_data = vehicle_data.iloc[start_idx:start_idx + self.window_size]

        features = window_data[FEATURE_COLS].values
        features_df = pd.DataFrame(features, columns=FEATURE_COLS)
        if self.scaler is not None:
            features = self.scaler.transform(features_df)
        else:
            features = features_df.values

        # 转换为tensor
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])

        return features, label


class LSTMModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # (batch_size, window_size, input_size)
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 分类
        output = self.classifier(lstm_out)

        return output


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.squeeze().to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Final Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Final Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('0422_gps/result_lstm/final_training_curves.png')
    plt.close()


def evaluate(model, dataloader, criterion, device):
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

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    plot_confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc


def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'High Speed', 'Acceleration', 'Tortuosity'],
                yticklabels=['Normal', 'High Speed', 'Acceleration', 'Tortuosity'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('0422_gps/result_lstm/confusion_matrix.png')
    plt.close()


# 主函数
def main():
    df = pd.read_csv("0422_gps/data/gps_features.csv")
    df = preprocess_data(df)

    unique_vids = df["vid_md5"].unique()
    train_vids, test_vids = train_test_split(unique_vids, test_size=0.2, random_state=42)
    val_vids, test_vids = train_test_split(test_vids, test_size=0.5, random_state=42)

    train_df = df[df["vid_md5"].isin(train_vids)]
    val_df = df[df["vid_md5"].isin(val_vids)]
    test_df = df[df["vid_md5"].isin(test_vids)]

    # 创建训练集的标准化器
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])
    
    # 保存标准化器
    joblib.dump(scaler, '0422_gps/result_lstm/scaler.gz')
    print("Scaler saved to 0422_gps/result_lstm/scaler.gz")

    # 创建数据集和数据加载器
    train_dataset = GPSDataset(train_df, scaler=scaler, is_train=True)
    val_dataset = GPSDataset(val_df, scaler=scaler, is_train=False)
    test_dataset = GPSDataset(test_df, scaler=scaler, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = len(FEATURE_COLS)
    num_classes = 4  # 0: 正常, 1: 高速, 2: 急加速/急减速, 3: 曲折行驶
    model = LSTMModel(input_size, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "0422_gps/result_lstm/best_model.pth")
            print("Saved best model")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load("0422_gps/result_lstm/best_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存标准化器
    joblib.dump(scaler, '0422_gps/result_lstm/scaler.gz')
    print("Scaler saved to 0422_gps/result_lstm/scaler.gz")


if __name__ == "__main__":
    main()
