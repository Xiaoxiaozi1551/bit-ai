import torch

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from dataset_generated import word_feature, DataSet
from model import MyLogisticRegression

# 初始化数据
from tqdm import tqdm


if __name__ == '__main__':
    word_data_dict, label_dict = word_feature()

    path = "./coref-dataset/"
    train_label_file = "./coref-dataset/train/"
    test_label_file = "./coref-dataset/test/"
    valid_label_file = "./coref-dataset/validation/"

    train_dataset = DataSet(path, train_label_file, word_data_dict, label_dict)
    test_dataset = DataSet(path, test_label_file, word_data_dict, label_dict)
    valid_dataset = DataSet(path, valid_label_file, word_data_dict, label_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    lr = 0.001
    num_iterations = 400
    model = MyLogisticRegression(learning_rate=lr, num_iterations=num_iterations).to(device)

    model.fit(train_dataset.X, train_dataset.y, valid_dataset.X, valid_dataset.y)

    test_pred = model.predict(test_dataset.X)

    print("lr:", lr, "\n", "Test accuracy:", f1_score(test_dataset.y, test_pred))
    print("Test classification_report:\n", classification_report(test_dataset.y, test_pred))

    model.plot()