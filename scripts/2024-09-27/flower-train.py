import torch
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入优化器模块
from torchvision import datasets, transforms, models  # 导入图像数据集、数据增强和预训练模型
from torch.utils.data import DataLoader, Subset  # 导入数据加载器和子集操作
from sklearn.model_selection import train_test_split  # 导入用于数据集划分的函数
from torchvision.models import ResNet101_Weights
from tabulate import tabulate
import os
import logging

logging.basicConfig(
    filename="flower-training.log",
    level=logging.INFO,
    format='%(message)s'
)

# current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 超参数设置
image_size = 224  # 将图片大小调整为 224x224
batch_size = 30  # 一次加载的图片数量
num_classes = 21  # 你的花朵分类任务有 14 个类别：康乃馨、鸢尾花、风铃草、金英花、玫瑰、落新妇、郁金香、金盏花、蒲公英、金鸡菊、黑眼菊、睡莲、向日葵、雏菊
epochs = 70  # 训练的周期数
learning_rate = 0.001  # 学习率

# 数据集路径
# train_data_dir = '/home/ai_cxp/cloud-model/dataset/flower/train'  # 训练数据集的文件夹路径
train_data_dir = os.path.join(current_dir, "dataset/flower", "train")  # 训练数据集的文件夹路径
# val_data_dir = '/home/ai_cxp/cloud-model/dataset/flower/val'  # 验证数据集的文件夹路径
val_data_dir = os.path.join(current_dir, "dataset/flower", "val")  # 验证数据集的文件夹路径
# data_dir = '../../data'  # 数据集的文件夹路径

# 数据预处理
# 这里定义了数据的变换，主要是调整大小、转换为张量和标准化
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整图片大小为 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化与ImageNet一致
])

train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_data_dir, transform=transform)
print(train_dataset.classes)
print(train_dataset.class_to_idx)

# 使用 DataLoader 加载数据，批量化加载，打乱训练集数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的 MobileNetV2 模型，并使用新版的 weights 参数
# MobileNet_V2_Weights.IMAGENET1K_V1 是从 ImageNet 数据集上训练的预训练权重
model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

# 冻结模型的所有参数，只训练最后的分类层
for param in model.parameters():
    param.requires_grad = False

# 修改模型的最后一层，适应当前3类云的分类任务
# MobileNetV2 的最后一层输出是 1280 个通道，我们将其输出设为 num_classes（即14类）
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 如果有可用的 GPU，将模型移动到 GPU 进行加速，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数（交叉熵损失）和优化器（Adam 优化器）
criterion = nn.CrossEntropyLoss()  # 分类问题中常用的损失函数

# 只更新最后一层的参数
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    results = []  # 存储每轮训练和验证结果
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()  # 设定模型为训练模式
        running_loss = 0.0  # 记录损失
        correct = 0  # 记录正确的预测数量
        total = 0  # 总计样本数量

        # 训练集
        # total_batches = len(train_loader)
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
            # progress = (batch_idx + 1) / total_batches * 100
            # print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{total_batches}], Progress: {progress:.2f}%")
            # logging.info(
            #     f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{total_batches}], Progress: {progress:.2f}%")

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

        # torch.save(model.state_dict(),
        #            '/home/ai_cxp/cloud-model/models/flower/2024-09-27/flower_recognition_model_batch.pth')
        torch.save(model.state_dict(),
                   os.path.join(current_dir, "models/flower/2024-09-27", "flower_recognition_model_batch.pth"))

        # 每轮训练完成后，打印当前轮次的结果
        print("\nResults after Epoch", epoch + 1)
        headers = ["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"]
        print(tabulate([results[-1]], headers=headers, tablefmt="grid"))
        logging.info(tabulate([results[-1]], headers=headers, tablefmt="grid"))

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(),
            #            '/home/ai_cxp/cloud-model/models/flower/2024-09-27/flower_recognition_model.pth')
            torch.save(model.state_dict(),
                       os.path.join(current_dir, "models/flower/2024-09-27", "flower_recognition_model_best.pth"))

    # 全部训练结束后，打印所有轮次的结果
    print("\nFinal Results after All Epochs:")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    logging.info(tabulate(results, headers=headers, tablefmt="grid"))


# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

# 保存模型
# torch.save(model.state_dict(), '/home/ai_cxp/cloud-model/models/flower/flower_recognition_model_2024-09-23.pth')
# torch.save(model.state_dict(), '../../models/cloud_recognition_model_2024-09-23.pth')
# print("模型已保存为 flower_recognition_model_2024-09-23.pth")
