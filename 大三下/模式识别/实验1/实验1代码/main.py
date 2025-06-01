import numpy as np
import os
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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

if __name__ == "__main__":
    # 数据集路径
    train_dataset_path = 'train'
    test_dataset_path = 'test'

    # 加载训练集和测试集
    X_train, y_train = load_images_from_folder(train_dataset_path)
    X_test, y_test = load_images_from_folder(test_dataset_path)

    # 手动实现PCA
    n_components = 50
    X_train_pca_manual, principal_components, mean = pca_manual(X_train, n_components)
    X_test_centered = X_test - mean
    X_test_pca_manual = transform_data(X_test_centered, principal_components)

    # 训练和评估KNN分类器
    knn = KNNClassifier(X_train_pca_manual, y_train, k=1)
    y_pred_manual = knn(X_test_pca_manual)
    accuracy_manual = accuracy_score(y_test, y_pred_manual)
    print(f'Accuracy : {accuracy_manual * 100:.2f}%')

    # 可视化前几个主成分
    plt.figure(figsize=(8, 6))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(principal_components[:, i].reshape((112, 92)), cmap='gray')
        plt.title(f'PC {i+1}')
        plt.axis('off')
    plt.show()
