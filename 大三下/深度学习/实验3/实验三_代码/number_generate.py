import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox
from PyQt5.QtGui import QPixmap
import torch
from torch.autograd import Variable
import numpy as np
import io
from PIL import Image
from gan import Generator, deprocess_img, NOISE_DIM  # 确保这些被正确导入
from convolutional_gan import ConvGenerator

class GANDigitGenerator(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()  # 初始化用户界面
        self.generator_type = "Generator"  # 默认选择 Generator 模型
        self.generator = self.load_generator()  # 加载初始生成器模型

    def initUI(self):
        self.setWindowTitle('GAN 数字生成器（芙宁娜倾情推荐）')  # 设置窗口标题

        # 创建组件
        self.label = QLabel('输入一个数字 (0-9):', self)  # 提示用户输入数字
        self.line_edit = QLineEdit(self)  # 输入框
        self.button = QPushButton('生成', self)  # 生成按钮
        self.image_label = QLabel(self)  # 用于显示生成的数字图片
        self.image_label.setFixedSize(280, 280)
        self.image_label.setStyleSheet("border: 1px solid black;")  # 添加边框以区分图片区域

        # 添加额外图片（可莉图片）
        self.extra_image_label = QLabel(self)  # 用于显示额外图片
        self.extra_image_label.setFixedSize(440, 280)
        self.extra_image_label.setStyleSheet("border: 1px solid black;")  # 添加边框以区分图片区域
        self.extra_image_label.setPixmap(QPixmap("fufu.jpg").scaled(440, 280))  # 加载并缩放图片

        # 添加下拉菜单，用于选择生成器模型
        self.model_combo_box = QComboBox(self)
        self.model_combo_box.addItem("Generator")
        self.model_combo_box.addItem("ConvGenerator")
        self.model_combo_box.currentIndexChanged.connect(self.update_generator_type)  # 连接下拉菜单的变化事件

        # 连接按钮的点击事件
        self.button.clicked.connect(self.generate_digit)

        # 布局
        hbox = QHBoxLayout()
        hbox.addWidget(self.extra_image_label)  # 添加额外图片区域到水平布局
        hbox.addWidget(self.image_label)  # 添加生成图片区域到水平布局

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)  # 添加标签到垂直布局
        vbox.addWidget(self.line_edit)  # 添加输入框到垂直布局
        vbox.addWidget(self.button)  # 添加按钮到垂直布局
        vbox.addWidget(self.model_combo_box)  # 添加下拉菜单到垂直布局
        vbox.addLayout(hbox)  # 添加水平布局到垂直布局

        self.setLayout(vbox)  # 设置窗口的主布局
        self.setGeometry(300, 300, 600, 400)  # 修改窗口大小以适应新图片区域

    def load_generator(self):
        # 根据选择加载不同的生成器模型
        if self.generator_type == "Generator":
            generator = Generator().cuda()  # 加载普通生成器
            generator.load_state_dict(torch.load('Generator.pth'))  # 加载预训练的模型参数
        else:
            generator = ConvGenerator().cuda()  # 加载卷积生成器
            generator.load_state_dict(torch.load('Conv_generator.pth'))  # 加载预训练的模型参数
        return generator

    def generate_digit(self):
        # 获取用户输入的数字，并生成对应的图像
        digit_str = self.line_edit.text()
        if not digit_str.isdigit() or not (0 <= int(digit_str) <= 9):
            QMessageBox.critical(self, '错误', '请输入0到9之间的数字')  # 显示错误消息
            return

        digit = int(digit_str)
        generated_image = self.generate_digit_image(self.generator, digit)  # 生成图像
        self.show_image(generated_image)  # 显示生成的图像

    def generate_digit_image(self, generator, digit, noise_size=NOISE_DIM):
        # 生成指定数字的图像
        generator.eval()  # 设置生成器为评估模式
        sample_noise = Variable(torch.randn(1, noise_size)).cuda()  # 生成随机噪声
        label = Variable(torch.LongTensor([digit])).cuda()  # 将输入的数字转换为张量
        with torch.no_grad():
            generated_image = generator(sample_noise, label).cpu().data.numpy()  # 生成图像
        return generated_image

    def show_image(self, image_array):
        # 显示生成的图像
        image = deprocess_img(image_array)  # 反处理图像
        image = (image * 255).astype(np.uint8).reshape(28, 28)  # 转换为8位图像
        image = Image.fromarray(image)  # 转换为PIL图像
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")  # 保存图像到缓冲区
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())  # 从缓冲区加载图像
        self.image_label.setPixmap(pixmap)  # 在标签中显示图像
        self.image_label.setScaledContents(True)  # 使图像适应标签大小

    def update_generator_type(self):
        # 更新生成器类型并重新加载生成器
        self.generator_type = self.model_combo_box.currentText()  # 获取当前选择的生成器类型
        self.generator = self.load_generator()  # 重新加载生成器

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GANDigitGenerator()  # 创建主窗口
    ex.show()  # 显示主窗口
    sys.exit(app.exec_())  # 进入应用程序主循环
