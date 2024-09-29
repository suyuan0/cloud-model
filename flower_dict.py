import os
from torchvision import datasets

current_dit = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(current_dit, 'dataset/flower', "val")

train_dataset = datasets.ImageFolder(train_dir)

class_names = train_dataset.classes
class_indices = train_dataset.class_to_idx

# 打印类别索引与名称对应关系
print("类别索引与名称对应关系:")
for class_name, index in class_indices.items():
    # print(f"索引: {index} -> 名称: {class_name} {len(os.listdir(os.path.join(train_dir, class_name)))}")
    print(len(os.listdir(os.path.join(train_dir, class_name))))


