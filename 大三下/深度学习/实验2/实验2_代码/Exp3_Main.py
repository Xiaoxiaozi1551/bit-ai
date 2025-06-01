"""
模型入口，程序执行的开始，在该文件中配置必要的训练步骤
"""
from matplotlib import pyplot as plt
from tqdm import tqdm

from Exp3_Config import Training_Config
from Exp3_DataSet import TextDataSet, TestDataSet
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN_Model
import torch


def train(model, loader, optimizer, loss_function):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    with tqdm(total=len(loader), desc="Training", unit="batch") as progress_bar:
        for index, data in enumerate(loader):
            optimizer.zero_grad()  # 梯度清零
            output = model(data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device))
            loss = loss_function(output, data[5].to(device))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()
            progress_bar.update(1)
    return total_loss / len(loader)


def validation(model, loader, loss_function):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct = 0
    with torch.no_grad():
        with tqdm(total=len(loader), desc="Validation", unit="batch") as progress_bar:
            for index, data in enumerate(loader):
                output = model(data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device),
                               data[4].to(device))
                loss = loss_function(output, data[5].to(device))  # 计算损失
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == data[5].to(device)).sum().item()
                progress_bar.update(1)
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


def predict(model, loader):
    model.eval()  # 设置模型为评估模式
    predictions = []
    with torch.no_grad():
        with tqdm(total=len(loader), desc="Predicting", unit="batch") as progress_bar:
            for index, data in enumerate(loader):
                output = model(data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device),
                               data[4].to(device))
                _, predicted = torch.max(output, 1)
                predictions.extend(predicted.tolist())
                progress_bar.update(1)
    return predictions


def plot_training_history(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # 绘制训练和验证损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = Training_Config()

    # 训练集验证集
    train_dataset = TextDataSet(filepath="./data/data_train.txt")
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)

    val_dataset = TextDataSet(filepath="./data/data_val.txt")
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size)

    # 测试集数据集和加载器
    test_dataset = TestDataSet("./data/test_exp3.txt")
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型对象
    Text_Model = TextCNN_Model(configs=config).to(device)
    # 损失函数设置
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
    # 优化器设置
    optimizer = torch.optim.Adam(params=Text_Model.parameters())  # torch.optim中的优化器进行挑选，并进行参数设置

    # 存储训练和验证过程中的损失和准确率
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 训练和验证
    for epoch in range(config.epoch):
        train_loss = train(Text_Model, loader=train_loader, optimizer=optimizer, loss_function=loss_function)
        val_loss, val_accuracy = validation(Text_Model, loader=val_loader, loss_function=loss_function)
        print(f"Epoch {epoch + 1}/{config.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 存储损失和准确率
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    # 预测（测试）
    predictions = predict(Text_Model, test_loader)
    with open('exp2_predict_labels_1120213587.txt', "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    plot_training_history(train_losses, val_losses, val_accuracies)
