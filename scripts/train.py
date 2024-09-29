import torch
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入优化器模块
from torchvision import datasets, transforms, models  # 导入图像数据集、数据增强和预训练模型
from torch.utils.data import DataLoader, Subset  # 导入数据加载器和子集操作
from sklearn.model_selection import train_test_split  # 导入用于数据集划分的函数
from torchvision.models import MobileNet_V2_Weights
from tabulate import tabulate
import os
import logging

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format='%(message)s'
)

# 超参数设置
image_size = 224  # 将图片大小调整为 224x224
batch_size = 32  # 一次加载的图片数量
num_classes = 11  # 你的云分类任务有 3 个类别：卷云、卷层云、卷积云、高积云、高层云、积云、积雨云、薄层云、层积云、层云、尾迹
epochs = 10  # 训练的周期数
learning_rate = 0.001  # 学习率

# 数据集路径
data_dir = '../data/'  # 数据集的文件夹路径

# 数据预处理
# 这里定义了数据的变换，主要是调整大小、转换为张量和标准化
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整图片大小为 224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化与ImageNet一致
])

# 加载整个数据集
# ImageFolder 会根据文件夹的结构自动分配标签
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

# 使用 train_test_split 将数据集划分为 80% 的训练集和 20% 的验证集
# stratify 参数确保数据划分时，类别的比例保持不变
train_indices, val_indices = train_test_split(
    list(range(len(full_dataset))), test_size=0.2, stratify=full_dataset.targets)

# 使用 Subset 创建训练集和验证集
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# 使用 DataLoader 加载数据，批量化加载，打乱训练集数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的 MobileNetV2 模型，并使用新版的 weights 参数
# MobileNet_V2_Weights.IMAGENET1K_V1 是从 ImageNet 数据集上训练的预训练权重
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

# 冻结模型的所有参数，只训练最后的分类层
for param in model.parameters():
    param.requires_grad = False

# 修改模型的最后一层，适应当前3类云的分类任务
# MobileNetV2 的最后一层输出是 1280 个通道，我们将其输出设为 num_classes（即3类）
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# 如果有可用的 GPU，将模型移动到 GPU 进行加速，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数（交叉熵损失）和优化器（Adam 优化器）
criterion = nn.CrossEntropyLoss()  # 分类问题中常用的损失函数

# 只更新最后一层的参数
optimizer = optim.Adam(model.classifier[1].parameters(), lr=learning_rate)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    results = []  # 存储每轮训练和验证结果

    for epoch in range(epochs):
        model.train()  # 设定模型为训练模式
        running_loss = 0.0  # 记录损失
        correct = 0  # 记录正确的预测数量
        total = 0  # 总计样本数量

        # 训练集
        total_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 将输入和标签加载到 GPU（如果可用）
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清除梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化参数

            running_loss += loss.item()  # 累计损失
            _, predicted = outputs.max(1)  # 取出预测的类别
            total += labels.size(0)  # 样本总数增加
            correct += predicted.eq(labels).sum().item()  # 预测正确的数量增加

            # 计算并显示进度百分比，保留两位小数
            progress = (batch_idx + 1) / total_batches * 100
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{total_batches}], Progress: {progress:.2f}%")
            logging.info(
                f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{total_batches}], Progress: {progress:.2f}%")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证集
        model.eval()  # 设定模型为评估模式
        val_loss = 0.0  # 验证集的损失
        val_correct = 0  # 验证集中正确预测的数量
        val_total = 0  # 验证集样本总数
        with torch.no_grad():  # 禁用梯度计算以加速验证过程
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()  # 累加验证集的损失
                _, predicted = outputs.max(1)  # 预测类别
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
