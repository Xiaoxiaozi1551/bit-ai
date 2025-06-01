import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载MNIST数据集
mnist = fetch_openml('mnist_784')
print("数据集大小：", mnist.data.shape)
print("标签集大小：", mnist.target.shape)
print("类别数量：", len(set(mnist.target)))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.astype(int).to_numpy(), dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.astype(int).to_numpy(), dtype=torch.long).to(device)


class AdaBoost:
    def __init__(self, n_clf=50):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.clfs = []
        self.alpha = []

        # 初始化权重
        w = torch.full((n_samples,), (1 / n_samples), dtype=torch.float32, device=device)

        for _ in range(self.n_clf):
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X.cpu().numpy(), y.cpu().numpy(), sample_weight=w.cpu().numpy())
            predictions = torch.tensor(clf.predict(X.cpu().numpy()), dtype=torch.float32, device=device)

            # 计算错误率和alpha值
            error = torch.sum(w * (predictions != y)) / torch.sum(w)
            if error > 0.5:
                continue  # 如果错误率大于0.5，则舍去该分类器
            alpha = 0.5 * torch.log((1.0 - error) / (error + 1e-10))

            # 更新权重
            w *= torch.exp(-alpha * y * predictions)
            w /= torch.sum(w)

            # 保存分类器和alpha值
            self.clfs.append(clf)
            self.alpha.append(alpha.item())

    def predict(self, X):
        clf_preds = [alpha * torch.tensor(clf.predict(X.cpu().numpy()), dtype=torch.float32, device=device) for
                     alpha, clf in zip(self.alpha, self.clfs)]
        y_pred = torch.sign(torch.sum(torch.stack(clf_preds), axis=0))
        return y_pred


# 使用OvR策略处理多分类问题
unique_classes = np.unique(y_train_tensor.cpu())
adaboost_models = {}

for cls in unique_classes:
    print(f"Training model for class {cls}")
    # 将当前类别设置为1，其他类别设置为-1
    y_train_binary = torch.where(y_train_tensor == cls, torch.tensor(1.0, device=device),
                                 torch.tensor(-1.0, device=device))

    # 训练Adaboost模型
    adaboost = AdaBoost(n_clf=50)
    adaboost.fit(X_train_tensor, y_train_binary)
    adaboost_models[cls] = adaboost


# 在测试集上进行预测
def predict_all(X):
    n_samples = X.shape[0]
    all_preds = torch.zeros((n_samples, len(unique_classes)), device=device)

    for cls, model in adaboost_models.items():
        preds = model.predict(X)
        all_preds[:, cls] = preds

    y_pred = torch.argmax(all_preds, axis=1)
    return y_pred


y_pred = predict_all(X_test_tensor)

# 计算准确率
accuracy = accuracy_score(y_test_tensor.cpu(), y_pred.cpu())
print(f'Adaboost 多分类测试准确率: {accuracy * 100:.2f}%')

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 绘制混淆矩阵
cm = confusion_matrix(y_test_tensor.cpu(), y_pred.cpu(), labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
disp.plot()
plt.show()

# 学习曲线
train_errors = []
test_errors = []
clf_range = range(1, 101, 5)

for n_clf in clf_range:
    adaboost_models = {}
    for cls in unique_classes:
        y_train_binary = torch.where(y_train_tensor == cls, torch.tensor(1.0, device=device),
                                     torch.tensor(-1.0, device=device))
        adaboost = AdaBoost(n_clf=n_clf)
        adaboost.fit(X_train_tensor, y_train_binary)
        adaboost_models[cls] = adaboost

    y_train_pred = predict_all(X_train_tensor)
    y_test_pred = predict_all(X_test_tensor)

    train_errors.append(1 - accuracy_score(y_train_tensor.cpu(), y_train_pred.cpu()))
    test_errors.append(1 - accuracy_score(y_test_tensor.cpu(), y_test_pred.cpu()))

plt.plot(clf_range, train_errors, label='训练误差')
plt.plot(clf_range, test_errors, label='测试误差')
plt.xlabel('分类器数量')
plt.ylabel('误差')
plt.legend()
plt.title('Adaboost学习曲线')
plt.show()


def plot_label_distribution(y_train, y_test):
    labels, counts_train = np.unique(y_train.cpu(), return_counts=True)
    labels, counts_test = np.unique(y_test.cpu(), return_counts=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(labels, counts_train, color='blue', alpha=0.7, label='训练集')
    ax[0].set_title('训练集标签分布')
    ax[0].set_xlabel('标签')
    ax[0].set_ylabel('数量')
    ax[0].legend()

    ax[1].bar(labels, counts_test, color='green', alpha=0.7, label='测试集')
    ax[1].set_title('测试集标签分布')
    ax[1].set_xlabel('标签')
    ax[1].set_ylabel('数量')
    ax[1].legend()

    plt.show()


plot_label_distribution(y_train_tensor, y_test_tensor)
