import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_dataset = datasets.ImageFolder(os.path.join(current_dir, "dataset/flower", "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(current_dir, "dataset/flower", "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


class FlowerCNN(nn.Module):
    def __init__(self):
        super(FlowerCNN, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 卷积层3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 根据输入图像大小调整
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 21)  # 21个花的种类

    def forward(self, x):
        # 前向传播
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 128 * 3 * 3)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # 使用log_softmax


# 创建模型
model = FlowerCNN()

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.NLLLoss()  # 使用负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()  # 累加损失
        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 计算正确预测数量

    epoch_loss = running_loss / len(train_loader)  # 平均损失
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


# 验证函数
def validate(model, val_loader, criterion):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            running_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 计算正确预测数量

    epochs_loss = running_loss / len(val_loader)  # 平均损失
    epochs_accuracy = correct / total  # 准确率
    return epochs_loss, epochs_accuracy


# 保存模型的函数
def save_model(model, epoch, is_best=False):
    model_path = os.path.join(current_dir, "models/flower/2024-09-29", f"model_epoch_${epoch}.pth")
    torch.save(model.state_dict(), model_path)
    if is_best:
        torch.save(model.state_dict(),
                   os.path.join(current_dir, "models/flower/2024-09-29", "best_model.pth"))  # 保存最佳模型


# 在训练循环中添加模型保存逻辑
def train_and_validate_with_saving(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_logs = []
    best_val_accuracy = 0.0  # 最佳验证准确率

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        train_logs.append({
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train Accuracy': train_accuracy,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_accuracy
        })

        # 打印日志
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2%}, "
              f"Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2%}")

        # 保存每个批次的模型
        save_model(model, epoch + 1)

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, epoch + 1, is_best=True)

    return train_logs


# 设置训练参数
num_epochs = 20  # 可以调整训练轮数

# 开始训练并保存模型
train_logs = train_and_validate_with_saving(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# 保存训练日志到CSV文件
log_df = pd.DataFrame(train_logs)
log_df.to_csv('train_logs.csv', index=False)
