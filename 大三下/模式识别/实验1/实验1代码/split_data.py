import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

# 原始数据集路径
dataset_path = 'ORL_dataset'
# 新的数据集路径
train_folder = 'train'
test_folder = 'test'

# 创建train和test文件夹
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# 遍历每个类的文件夹
for label in os.listdir(dataset_path):
    subfolder = os.path.join(dataset_path, label)
    if os.path.isdir(subfolder):
        images = os.listdir(subfolder)
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42, shuffle=True)

        # 创建每个类的子文件夹
        train_class_folder = os.path.join(train_folder, label)
        test_class_folder = os.path.join(test_folder, label)
        if not os.path.exists(train_class_folder):
            os.makedirs(train_class_folder)
        if not os.path.exists(test_class_folder):
            os.makedirs(test_class_folder)

        # 将图像移动到训练集文件夹
        for img in train_images:
            img_path = os.path.join(subfolder, img)
            shutil.copy(img_path, os.path.join(train_class_folder, img))

        # 将图像移动到测试集文件夹
        for img in test_images:
            img_path = os.path.join(subfolder, img)
            shutil.copy(img_path, os.path.join(test_class_folder, img))

print("数据集分割完成！")
