import os
import shutil
import random

dataset_dir = 'data/'

train_dir = "dataset/cloud/train_data/"
val_dir = "dataset/cloud/val_data/"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset(dataset_dir, train_dir, val_dir, split_ratio=0.2):
    classes = os.listdir(dataset_dir)

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_dir)
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        val_images = images[:split_point]
        train_images = images[split_point:]

        create_dir(os.path.join(train_dir, class_name))
        create_dir(os.path.join(val_dir, class_name))

        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copyfile(src, dst)

        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copyfile(src, dst)
    print("数据集已划分并复制到指定文件夹。")

split_dataset(dataset_dir, train_dir, val_dir)