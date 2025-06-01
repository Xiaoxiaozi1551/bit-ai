import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, roc_curve, \
    auc
from torch import nn
from tqdm import tqdm

losses = []  # 用于记录损失值
valid_losses = []
F1_list = []
accuracy_list = []
# precision_list = []
# recall_list = []


class MyLogisticRegression(nn.Module):
    def __init__(self, learning_rate=0.0005, num_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.epsilon = 1e-8  # 用于避免除零错误的小值
        # self.gradient_accumulation = self.learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, valid_X, valid_y):
        X = np.array(X)
        y = np.array(y)
        valid_X = np.array(valid_X)
        valid_y = np.array(valid_y)
        # for i in range(y.shape[0]):
        #     print(y[i])
        num_samples, num_features = X.shape
        # print(X.shape)

        self.weights = np.zeros(num_features)
        self.bias = 0
        self.gradient_accumulation = np.zeros(num_features)  # 初始化累积梯度为零向量

        fpr_list = []
        tpr_list = []
        auc_list = []

        # 循环
        for _ in tqdm(range(self.num_iterations), desc="训练", unit="次"):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            loss = np.mean(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))  # 计算交叉熵损失
            # y_ = torch.tensor(y)
            # y_pred_ = torch.tensor(y_pred)
            # loss = criterion(y_pred_, y_)

            losses.append(loss)  # 记录损失值

            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.gradient_accumulation += dw ** 2  # 累加梯度的平方和

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 验证集
            valid_linear_model = np.dot(valid_X, self.weights) + self.bias
            valid_pred = self._sigmoid(valid_linear_model)

            valid_loss = np.mean(-(valid_y * np.log(valid_pred) + (1 - valid_y) * np.log(1 - valid_pred)))  # 计算交叉熵损失
            # valid_pred = self.predict(valid_X)
            valid_pred = np.where(valid_pred > 0.5, 1, 0)
            valid_losses.append(valid_loss)
            f1 = f1_score(valid_y, valid_pred)
            accuracy = accuracy_score(valid_y, valid_pred)
            F1_list.append(f1)
            # valid_losses.append(valid_loss)
            accuracy_list.append(accuracy)

            # Compute false positive rate, true positive rate, and thresholds
            fpr, tpr, thresholds = roc_curve(valid_y, valid_pred)

            # Calculate area under the ROC curve
            roc_auc = auc(fpr, tpr)

            # Append values to lists
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)

        # 输出损失值
        sum_pos = 0
        sum_neg = 0
        for item in y:
            if item == 1:
                sum_pos += 1
            else:
                sum_neg += 1
        print(sum_neg / sum_pos)
        print("Train loss 历史记录:", losses)
        print("Valid loss 历史记录:", valid_losses)
        print("F1-score历史记录:", F1_list)
        valid_pred = self.predict(valid_X)
        f1 = f1_score(valid_y, valid_pred)
        precision = precision_score(valid_y, valid_pred)
        recall = recall_score(valid_y, valid_pred)
        print(classification_report(valid_y, valid_pred))
        print("Validation F1:", f1)
        print("Validation precision:", precision)
        print("Validation recall:", recall)
        print("Validation accuracy:", accuracy_score(valid_y, valid_pred))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_class = np.where(y_pred > 0.5, 1, 0)
        return y_pred_class

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # 绘制损失函数曲线
        # ax1.set_xticks(losses)
        ax1.plot(losses, label='Train_loss')
        ax1.plot(valid_losses, label='Valid_loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()

        # 绘制验证准确率曲线
        # ax2.set_xticks(accuracy_epochs)
        ax2.plot(F1_list, label='F1_score')
        ax2.plot(accuracy_list, label='Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1_score')
        ax2.set_title('Validation F1 and accuracy')
        ax2.legend()

        # 调整子图间的间距
        fig.subplots_adjust(hspace=0.5)

        fig.suptitle('lr = '+str(self.learning_rate))

        # 显示图像
        plt.show()