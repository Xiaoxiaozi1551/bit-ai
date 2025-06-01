import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QVBoxLayout, QWidget, QHBoxLayout, QGridLayout)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import matplotlib.pyplot as plt
import torch

# 定义一些全局变量以存储训练集信息
principal_components = None
mean = None
knn = None

# 加载图像
def load_images_from_folder(folder, image_size=(112, 92)):
    images = []
    labels = []
    for label in os.listdir(folder):
        subfolder = os.path.join(folder, label)
        if os.path.isdir(subfolder):
            for filename in os.listdir(subfolder):
                img_path = os.path.join(subfolder, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                img = np.array(img).flatten()
                images.append(img)
                labels.append(int(label[1:]))  # 文件夹名为 's1', 's2' 等
    return np.array(images), np.array(labels)

# 数据标准化
def standardize_data(X):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    return X_centered, mean

# 计算协方差矩阵
def compute_covariance_matrix(X):
    return np.cov(X, rowvar=False)

# 计算特征值和特征向量
def compute_eigenvalues_and_eigenvectors(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

# 选择主要成分
def select_principal_components(eigenvalues, eigenvectors, n_components):
    return eigenvectors[:, :n_components]

# 转换数据
def transform_data(X, principal_components):
    return np.dot(X, principal_components)

# 主动实现PCA
def pca_manual(X, n_components):
    X_centered, mean = standardize_data(X)
    cov_matrix = compute_covariance_matrix(X_centered)
    eigenvalues, eigenvectors = compute_eigenvalues_and_eigenvectors(cov_matrix)
    principal_components = select_principal_components(eigenvalues, eigenvectors, n_components)
    X_pca = transform_data(X_centered, principal_components)
    return X_pca, principal_components, mean

# KNN分类器
class KNNClassifier(torch.nn.Module):
    def __init__(self, X_train, y_train, k=1):
        super(KNNClassifier, self).__init__()
        self.X_train = torch.tensor(X_train, dtype=torch.float32).cuda()
        self.y_train = torch.tensor(y_train).cuda()
        self.k = k

    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        distances = torch.cdist(X, self.X_train)
        knn_indices = distances.topk(self.k, largest=False).indices
        knn_labels = self.y_train[knn_indices]
        y_pred = torch.mode(knn_labels, dim=1).values
        return y_pred.cpu().numpy()

# 加载训练数据并初始化PCA和KNN模型
def initialize_model():
    global principal_components, mean, knn
    n_components = 50
    train_dataset_path = 'train'
    X_train, y_train = load_images_from_folder(train_dataset_path)
    X_train_pca_manual, principal_components, mean = pca_manual(X_train, n_components)
    knn = KNNClassifier(X_train_pca_manual, y_train, k=1)

# PyQt应用程序类
class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Face Recognition with PCA')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.upload_button = QPushButton('上传图片')
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        self.pca_images_layout = QHBoxLayout()
        self.layout.addLayout(self.pca_images_layout)

        self.pca_labels = [QLabel() for _ in range(5)]
        for label in self.pca_labels:
            label.setAlignment(Qt.AlignCenter)
            self.pca_images_layout.addWidget(label)

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "上传图片", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))

            img = Image.open(file_path).convert('L')
            img = img.resize((112, 92))
            img = np.array(img).flatten()
            img_centered = img - mean
            img_pca = transform_data(img_centered.reshape(1, -1), principal_components)
            pred = knn(img_pca)

            self.result_label.setText(f'预测分类: {pred[0]}')

            plt.figure(figsize=(8, 6))
            for i in range(5):
                plt.subplot(1, 5, i + 1)
                plt.imshow(principal_components[:, i].reshape((112, 92)), cmap='gray')
                plt.title(f'PC {i + 1}')
                plt.axis('off')
            pca_image_path = os.path.join('pca_components.png')
            plt.savefig(pca_image_path)
            plt.close()

            for i in range(5):
                self.pca_labels[i].setPixmap(QPixmap(pca_image_path).scaled(112, 92, Qt.KeepAspectRatio))

if __name__ == '__main__':
    initialize_model()
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
