import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

loss_values = []  # 用于存储每个epoch的损失值
loss_epochs = []
accuracy_values = []  # 用于存储每个epoch的验证准确率
accuracy_epochs = []


# 数据加载
class CifarDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        # 添加数据集的初始化内容
        self.img_dir = img_dir
        self.samples = self.load_samples(label_file)
        self.transfomer = transform

    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        item = self.samples[index]
        img_path = os.path.join(self.img_dir, item.split()[0])
        img = Image.open(img_path).convert('RGB')
        if self.transfomer is not None:
            img = self.transfomer(img)
        img = torch.tensor(np.array(img))  # 将PIL.Image.Image转换为torch.Tensor
        if len(item.split()) < 2:
            return img, -1  # 利用-1当占位符
        label = int(item.split()[1])
        return img, label

    def __len__(self):
        # 添加len函数的相关内容
        return len(self.samples)

    def load_samples(self, label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            samples = [line.strip() for line in lines]
        return samples


# 构建模型_全连接网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义模型的网络结构
        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 定义模型前向传播的内容
        x = x.view(x.size(0), -1)
        x = x.to(self.fc1.weight.dtype)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 调整学习率函数
def adjust_learning_rate(optimizer, current_step, total_steps):
    """根据当前步数调整学习率"""
    lr = 0.0001  # 初始学习率
    min_lr = 0.00001  # 最小学习率
    warmup_steps = 500  # 预热步数
    cosine_decay_steps = total_steps - warmup_steps  # 余弦退火衰减步数

    if current_step < warmup_steps:
        lr = lr * (current_step / warmup_steps)  # 预热阶段，学习率线性增加
    else:
        progress = (current_step - warmup_steps) / cosine_decay_steps  # 余弦退火进度
        lr = min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * progress))  # 余弦退火调整学习率

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# 定义 train 函数
def train():
    net.train()
    # 参数设置
    epoch_num = 30
    val_num = 2
    total_steps = len(train_loader) * epoch_num  # 计算总的训练步数

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
            for i, (x, y) in enumerate(train_loader, 0):
                images, labels = x.to(device), y.to(device)
                optimizer.zero_grad()
                # Forward
                outputs = net(images)
                # Backward
                loss = criterion(outputs, labels)
                loss.backward()
                # Update
                optimizer.step()

                # 学习率调整
                current_step = epoch * len(train_loader) + i
                adjust_learning_rate(optimizer, current_step, total_steps)

                running_loss += loss.item()
                pbar.set_postfix({'Loss': running_loss / (i + 1)})
                pbar.update()

        loss_values.append(loss.item())  # 将损失值添加到列表中
        loss_epochs.append(epoch+1)

        # 模型训练n轮之后进行验证
        if epoch % val_num != 0:
            # print(epoch)
            accuracy_epochs.append(epoch+1)
            validation()

    print('Finished Training!')


# 定义 validation 函数
def validation():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(dev_loader), desc='Validation', unit='batch') as pbar:
            for (x, y) in dev_loader:
                images, labels = x.to(device), y.to(device)
                # 在这一部分撰写验证的内容
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.update()

    accuracy = correct / total
    accuracy_values.append(accuracy)  # 将验证准确率添加到列表中
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)


# 定义 test 函数
def test():
    # 将预测结果写入result.txt文件中，格式参照实验1
    net.eval()
    with torch.no_grad():
        with open('result.txt', 'w') as f:
            for (x, y) in test_loader:
                image, _ = x.to(device), y.to(device)
                outputs = net(image)
                _, predicted = torch.max(outputs.data, 1)
                for i in predicted:
                    str_ = str(i.item()) + '\n'
                    f.write(str_)


if __name__ == "__main__":
    # 数据集路径
    img_dir = "./dataset/image"
    train_label_file = "./dataset/trainset.txt"
    dev_label_file = "./dataset/validset.txt"
    test_label_file = "./dataset/testset.txt"

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
         transforms.RandomGrayscale(),
         transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    # 构建数据集
    train_set = CifarDataset(img_dir, train_label_file, transform=transform)
    dev_set = CifarDataset(img_dir, dev_label_file, transform=transform)
    test_set = CifarDataset(img_dir, test_label_file, transform=transform)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, batch_size=128)
    test_loader = DataLoader(dataset=test_set, batch_size=128)

    # 初始化模型对象
    net = Net().to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)

    # 模型训练
    train()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # 绘制损失函数曲线
    ax1.set_xticks(loss_epochs)
    ax1.plot(loss_epochs, loss_values)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')

    # 绘制验证准确率曲线
    ax2.set_xticks(accuracy_epochs)
    ax2.plot(accuracy_epochs, accuracy_values)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')

    # 调整子图间的间距
    fig.subplots_adjust(hspace=0.5)

    # 显示图像
    plt.show()

    # 对模型进行测试，并生成预测结果
    test()
