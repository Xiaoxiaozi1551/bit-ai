#! ~/opt/anaconda3/bin/python3.8
import numpy as np
import random
import argparse

"""
逻辑回归模型
请在pass处按注释要求插入代码以完善模型功能
"""

class LogisticRegression:
    def __init__(self, word_dim=300, max_len=80, learning_rate=0.0001, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # 若输入为词向量求和的形式，请注意修改权重矩阵的大小为正确的值
        self.weights = np.random.randn(max_len * word_dim)

        self.bias = random.uniform(-0.1, 0.1)
        # 模型参数保存路径：
        self.weights_path = "./weights.npy"
        self.bias_path = "./bias.npy"

    def sigmoid(self, x):
        # 请实现sigmoid函数
        return 1.0 / (1 + np.exp(-x))

    def loss(self, out, label):
        # 请实现2分类的 cross entropy loss
        loss = np.sum(np.nan_to_num(-label*np.log(out + 1e-10)-(1-label)*np.log(1-out + 1e-10)))
        return loss

    def forward(self, X):
        """
        正向传播
        :param X: 模型输入，X第一维为batch_size，第二维为输入向量
        :return: 模型输出
        """
        y_hat = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return y_hat

    def gradient_descent(self, X, out, y):
        """
        利用梯度下降调整参数。根据推导出的梯度公式，更新self.weights和self.bias
        :param X: 模型输入
        :param out: 模型输出
        :param y: label值
        :return: None
        """
        error = out - np.array(y)
        d_weights = np.dot(X.T, error)
        d_bias = np.sum(error)
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

    def train(self, train_iter, test_set):
        """
        根据训练数据和测试数据训练模型，并在每个epoch之后计算损失
        :param train_iter: 训练数据迭代器
        :param test_set: 测试数据集
        :return: 每个epoch的训练损失和测试损失
        """
        train_losses = []  # 记录平均训练损失
        test_losses = []  # 记录平均测试损失
        for epoch in range(self.epochs):
            train_loss = 0
            n_samples = 0
            for data, label in train_iter:  # batch_size就是16
                """
                正向传播
                调用self.gradient_descent()来更新参数
                记录训练损失
                """
                out = self.forward(data)
                self.gradient_descent(data, out, label)
                train_loss += self.loss(out, label)
                n_samples += len(data)
            # 计算损失
            train_loss /= n_samples  # 所有样本上的训练损失
            test_loss = self.test(test_set)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print("epoch{}/{} training loss:{}, test loss:{}".format(epoch, self.epochs, train_loss, test_loss))

        # 保存模型参数
        np.save(self.weights_path, self.weights)
        np.save(self.bias_path, self.bias)

        return train_losses, test_losses

    def test(self, test_set):
        """
        计算平均测试损失
        :param test_set: 测试集
        :return: 测试集损失
        """
        test_loss = 0
        n_samples = len(test_set)
        for data, label in test_set:
            """
            计算测试集总损失
            """
            out = self.forward(data)
            test_loss += self.loss(out, label)

        test_loss /= n_samples
        return test_loss

    def test_accuracy(self, test_set):
        """
        测试模型分类精度
        :param test_set: 测试集
        :return: 模型精度（百分数）
        """
        rights = 0  # 分类正确的个数
        n_samples = len(test_set)
        for data, label in test_set:
            """
            记录分类正确的样本个数
            """
            out = self.forward(data)
            predictions = (out > 0.5).astype(int)
            rights += np.sum(predictions == label)

        return (rights / n_samples) * 100
