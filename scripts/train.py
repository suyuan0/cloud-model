import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.models import MobileNet_V2_Weights
from tabulate import tabulate
import os
import logging

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format='%(message)s'
)

# 图片大小和批量大小
image_size = 224
batch_size = 32
num_classes = 11  # 你的云分类任务有 3 个类别：卷云、卷层云、卷积云、高积云、高层云、积云、积雨云、薄层云、层积云、层云、尾迹
epochs = 1
learning_rate = 0.001

# 数据集路径
data_dir = '../data/'

# 定义数据增强与预处理
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整图片大小为 224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化与ImageNet一致
])

# 加载整个数据集
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

# 使用 train_test_split 将数据划分为训练集和验证集，按 80:20 的比例
train_indices, val_indices = train_test_split(
    list(range(len(full_dataset))), test_size=0.2, stratify=full_dataset.targets)

# 使用 Subset 创建训练集和验证集
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的 MobileNetV2 模型
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

# 冻结预训练模型的所有参数
for param in model.parameters():
    param.requires_grad = False

# 修改模型的最后一层，适应3类分类任务
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 仅优化最后一层的参数
optimizer = optim.Adam(model.classifier[1].parameters(), lr=learning_rate)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    results = []  # 存储每轮训练和验证结果

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练集
        total_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 计算并显示进度百分比，保留两位小数
            progress = (batch_idx + 1) / total_batches * 100
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{total_batches}], Progress: {progress:.2f}%")
            logging.info(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{total_batches}], Progress: {progress:.2f}%")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证集
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # 将当前 epoch 的数据存入 results 列表
        results.append(
            [f"{epoch + 1}/{epochs}", f"{train_loss:.2f}", f"{train_acc:.2f}%", f"{val_loss:.2f}", f"{val_acc:.2f}%"])

        # 每轮训练完成后，打印当前轮次的结果
        print("\nResults after Epoch", epoch + 1)
        headers = ["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"]
        print(tabulate([results[-1]], headers=headers, tablefmt="grid"))
        logging.info(tabulate([results[-1]], headers=headers, tablefmt="grid"))

    # 全部训练结束后，打印所有轮次的结果
    print("\nFinal Results after All Epochs:")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    logging.info(tabulate(results, headers=headers, tablefmt="grid"))


# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

# 保存模型
torch.save(model.state_dict(), '../models/cloud_recognition_model.pth')
print("模型已保存为 cloud_recognition_model.pth")
