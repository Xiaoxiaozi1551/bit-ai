import math

import numpy as np

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn.preprocessing import StandardScaler


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout=0.1),
            num_layers,
        )
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  # 只使用最后一个时间步的输出作为预测
        return x


def train():
    model.train()

    epoch_num = 200

    for epoch in range(epoch_num):
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
            for i, (X, y) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                # Forward
                outputs = model(X)
                # Backward
                loss = criterion(outputs, y)
                loss.backward()
                # Update
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'Loss': running_loss / (i + 1)})
                pbar.update()


if __name__ == '__main__':
    # 数据处理
    data_ha = []
    # elements = ['工业企业研发费用增速']
    data_all = pd.read_excel('实验数据.xlsx', skiprows=[1])

    # 时间戳
    data_all['year'] = data_all['年度季度缩写'].astype(str).str[:4]
    # print(data_all['year'])

    # elements = ['year',
    #             '季度',
    #             '大中型重点企业研发费用规模',
    #             '工业企业研发费用规模',
    #             '工业企业研发费用增速',
    #             '工业企业研发费用占比',
    #             'GDP',
    #             '技术市场成交额',
    #             '大中型工业上市企业营业总收入',
    #             '大中型工业上市企业营业收入',
    #             '大中型工业上市企业营业总成本',
    #             '大中型工业上市企业研发费用',
    #             '大中型工业上市企业营业利润',
    #             '大中型工业上市企业利润总额',
    #             '大中型工业上市企业净利润',
    #             ]
    # elements = ['year',
    #             '季度',
    #             '大中型重点企业研发费用规模',
    #             '软信业企业研发费用规模',
    #             '软信业企业研发费用增速',
    #             '软信业企业研发费用占比',
    #             'GDP',
    #             '技术市场成交额',
    #             '大中型软信业上市企业营业总收入',
    #             '大中型软信业上市企业营业收入',
    #             '大中型软信业上市企业营业总成本',
    #             '大中型软信业上市企业研发费用',
    #             '大中型软信业上市企业营业利润',
    #             '大中型软信业上市企业利润总额',
    #             '大中型软信业上市企业净利润',
    #             ]
    elements = ['year',
                '季度',
                '大中型重点企业研发费用规模',
                '科技服务业企业研发费用规模',
                '科技服务业研发费用增速',
                '科技服务业企业研发费用占比',
                'GDP',
                '技术市场成交额',
                '大中型科技服务业上市企业营业总收入',
                '大中型科技服务业上市企业营业收入',
                '大中型科技服务业上市企业营业总成本',
                '大中型科技服务业上市企业研发费用',
                '大中型科技服务业上市企业营业利润',
                '大中型科技服务业上市企业利润总额',
                '大中型科技服务业上市企业净利润',
                ]
    length = len(data_all)
    for index, element in enumerate(elements):
        data_element = data_all[element].values.astype(np.float64)
        data_element = data_element.reshape(length, 1)
        # print(data_element)
        data_ha.append(data_element)

    x_hat = np.concatenate(data_ha, axis=1)  # 同一季度的数据放一起

    seq_len = 6  # 预测步长
    y = x_hat[seq_len:, 4]
    y = y.reshape(y.shape[0], 1)

    X_data = []
    for i in range(14 - seq_len):
        temp = x_hat[i:i + seq_len]
        temp = temp.reshape(-1)
        X_data.append(temp)

    X_ = np.array(X_data)
    X = torch.from_numpy(X_)  # 得到每行的特征数据

    # 归一化
    ss_x = StandardScaler()
    ss_y = StandardScaler()

    X = ss_x.fit_transform(X)
    y = ss_y.fit_transform(y)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    X = X.type(torch.float32)
    y = y.type(torch.float32)
    y = y.reshape(y.shape[0], 1)
    # print(X)

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, shuffle=False)

    input_dim = X.shape[1]  # 输入维度
    output_dim = 1  # 输出维度（预测研发费用增速）
    hidden_dim = 128  # 隐藏层维度
    num_layers = 5  # Transformer编码器层数
    num_heads = 6  # 注意力头数

    model = TransformerModel(input_dim, output_dim, hidden_dim, num_layers, num_heads)
    # model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)

    train()

    # 在训练后进行预测
    model.eval()
    with torch.no_grad():
        predicted_values = model(X)
    predict = ss_y.inverse_transform(predicted_values)
    # predicted_values = predicted_values.squeeze().numpy()

    y = ss_y.inverse_transform(y)
    predict_ = torch.from_numpy(predict)
    y_ = torch.from_numpy(y)

    print(criterion(predict_, y_).item())

    # 未来6个季度
    X_future1 = X[4:]
    X_future2 = X[:2]
    X_future = torch.cat((X_future1, X_future2), dim=0)
    with torch.no_grad():
        predicted_future = model(X_future)
    predicted_future_ = ss_y.inverse_transform(predicted_future)
    print(predicted_future_)

    # 绘制预测值和真实值的折线图
    mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示中文字体
    mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='真实值')
    plt.plot(predict, label='预测值')
    plt.plot([8, 9, 10, 11, 12, 13], predicted_future_, label='未来值')
    plt.xlabel('季度')
    plt.ylabel('研发费用增速')
    plt.title('预测值和真实值的折线图')
    plt.legend()
    plt.show()
