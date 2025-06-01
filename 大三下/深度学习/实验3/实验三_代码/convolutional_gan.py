import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import random

import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision.datasets import MNIST

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置随机种子以保证结果的可重复性
SEED = 3587
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# 定义噪声维度和训练参数
NOISE_DIM = 96
NUM_EPOCHS = 30
batch_size = 128

# 显示生成的图像
def show_images(images):
    images = deprocess_img(images)  # 反处理图像以恢复到正常范围
    grid_size = int(np.sqrt(images.shape[0]))  # 计算网格大小
    fig = plt.figure(figsize=(grid_size, grid_size))
    gs = gridspec.GridSpec(grid_size, grid_size, wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticks([])  # 移除x轴刻度
        ax.set_yticks([])  # 移除y轴刻度
        ax.set_aspect('equal')
        plt.imshow(img.reshape((28, 28)), cmap='gray')  # 显示灰度图像

    plt.show()

# 图像预处理函数
def preprocess_img(x):
    x = tfs.ToTensor()(x)  # 将图像转换为张量
    return (x - 0.5) / 0.5  # 将像素值归一化到[-1, 1]

# 图像反处理函数
def deprocess_img(x):
    return (x + 1.0) / 2.0  # 将像素值恢复到[0, 1]

# 加载MNIST数据集
train_set = torchvision.datasets.MNIST('./mnist', train=True, download=True, transform=preprocess_img)
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 定义卷积判别器网络
class ConvDiscriminator(nn.Module):
    def __init__(self):
        super(ConvDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)  # 标签嵌入层
        self.conv = nn.Sequential(
            nn.Conv2d(1 + 10, 32, 5, 1),  # 输入通道数为1（图像）+10（标签）
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),  # 计算得到的 64 * 4 * 4 是卷积层输出的特征图大小
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)  # 获取标签的嵌入表示
        c = c.unsqueeze(2).unsqueeze(3)  # 增加维度以便与图像特征拼接
        c = c.expand(-1, -1, x.size(2), x.size(3))  # 扩展标签嵌入的维度
        x = torch.cat([x, c], dim=1)  # 将图像特征和标签嵌入进行拼接
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

# 定义卷积生成器网络
class ConvGenerator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super(ConvGenerator, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)  # 标签嵌入层
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 10, 1024),  # 噪声向量与标签嵌入进行拼接
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)  # 获取标签的嵌入表示
        x = torch.cat([x, c], dim=1)  # 将噪声向量和标签嵌入进行拼接
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)  # 7x7是因为我们要得到和MNIST一样大小的图像
        x = self.conv(x)
        return x

# 定义损失函数
bce_loss = nn.BCEWithLogitsLoss()

def discriminator_loss(logits_real, logits_fake):
    size = logits_real.size(0)
    true_labels = Variable(torch.ones(size, 1)).cuda()  # 真实样本标签为1
    false_labels = Variable(torch.zeros(size, 1)).cuda()  # 生成样本标签为0
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)  # 计算判别器损失
    return loss

def generator_loss(logits_fake):
    size = logits_fake.size(0)
    true_labels = Variable(torch.ones(size, 1)).cuda()  # 生成样本标签为1
    loss = bce_loss(logits_fake, true_labels)  # 计算生成器损失
    return loss

def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))  # Adam优化器
    return optimizer

# 训练GAN模型
def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, noise_size=NOISE_DIM,
                num_epochs=NUM_EPOCHS):
    for epoch in range(num_epochs):
        for x, labels in train_data:
            D_net.train()  # 训练模式
            G_net.train()  # 训练模式

            bs = x.size(0)
            real_data = Variable(x).cuda()  # 获取真实数据
            labels = Variable(labels).cuda()  # 获取标签

            logits_real = D_net(real_data, labels)  # 判别器判别真实数据

            sample_noise = Variable(torch.randn(bs, noise_size)).cuda()  # 生成随机噪声
            fake_images = G_net(sample_noise, labels)  # 生成假数据

            logits_fake = D_net(fake_images.detach(), labels)  # 判别器判别假数据，detach用于阻止梯度传播到生成器

            d_total_error = discriminator_loss(logits_real, logits_fake)  # 计算判别器总损失

            D_optimizer.zero_grad()  # 清空梯度
            d_total_error.backward()  # 反向传播
            D_optimizer.step()  # 更新判别器参数

            logits_fake = D_net(fake_images, labels)  # 重新计算假数据的判别结果

            g_error = generator_loss(logits_fake)  # 计算生成器损失

            G_optimizer.zero_grad()  # 清空梯度
            g_error.backward()  # 反向传播
            G_optimizer.step()  # 更新生成器参数

        print(f'第 {epoch + 1} 轮训练，判别器损失: {d_total_error.item():.4f}, 生成器损失: {g_error.item():.4f}')

        if epoch % 5 == 4:  # 每5轮展示一次生成结果
            G_net.eval()  # 评估模式
            sample_noise = Variable(torch.randn(25, noise_size)).cuda()  # 生成25个随机噪声
            fake_labels = Variable(torch.randint(0, 10, (25,))).cuda()  # 生成25个随机标签
            fake_images = G_net(sample_noise, fake_labels)  # 生成假数据
            show_images(fake_images.cpu().data.numpy())  # 展示生成的图像

    # 保存生成器模型
    torch.save(G_net.state_dict(), 'Conv_generator.pth')
    print("模型已保存")

if __name__ == '__main__':
    # 初始化判别器和生成器
    D = ConvDiscriminator().cuda()
    G = ConvGenerator().cuda()

    # 获取优化器
    D_optim = get_optimizer(D)
    G_optim = get_optimizer(G)

    # 训练GAN模型
    train_a_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss)
