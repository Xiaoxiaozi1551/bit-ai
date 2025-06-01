import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataSet import Corpus
from My_Model import TransformerModel


def train():
    max_valid_bleu = 0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Training loop
        for batch_x, batch_y in tqdm(data_loader_train, desc=f"Epoch {epoch} Training"):
            optimizer.zero_grad()  # Zero the gradients

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            output = model(batch_x, batch_y)
            loss = loss_function(output.view(-1, output.size(-1)), batch_y.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            correct = (predictions == batch_y).sum().item()
            total_correct += correct
            total_samples += batch_y.numel()

        # Calculate training accuracy
        train_acc = total_correct / total_samples

        # Validation
        valid_bleu = valid()

        if valid_bleu > max_valid_bleu:
            # Save the model when validation BLEU improves
            torch.save(model.state_dict(), os.path.join(output_folder, "model.ckpt"))
            max_valid_bleu = valid_bleu

        print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc * 100:.2f}%, "
              f"valid BLEU: {valid_bleu * 100:.2f}%")

def valid():
    model.eval()  # Set the model to evaluation mode
    references = []  # Ground truth sentences
    hypotheses = []  # Predicted sentences

    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader_valid, desc="Validation"):
            # Forward pass
            output__ = model(batch_x.to(device), batch_x.to(device))
            output = first_sentence = model.beam_search_decoding(output__)  # 第一句话
            for i in range(3):
                output_ = model(first_sentence.to(device), first_sentence.to(device))
                next_sentence = model.beam_search_decoding(output_)
                output = torch.cat((output, next_sentence), dim=1)
                first_sentence = next_sentence

            # Convert indices to words
            reference = [[dataset.dictionary.tkn2word[idx] for idx in sent if idx != dataset.dictionary.word2tkn['[PAD]']] for sent in batch_y.tolist()]

            references.extend(reference)
            # 外部循环迭代每个样本
            for sample_output in output:
                hypothesis = []  # 存储当前样本的假设序列
                # 内部循环迭代每个时间步
                for idx in sample_output.squeeze().tolist():
                    if idx != dataset.dictionary.word2tkn['[PAD]']:
                        word = dataset.dictionary.tkn2word[idx]
                        hypothesis.append(word)
                # 将当前样本的假设序列添加到总列表中
                hypotheses.append(hypothesis)

        # Calculate BLEU score
        smoothie = SmoothingFunction().method1
        bleu_score = corpus_bleu(references, hypotheses, weights=(0.5, 0.5), smoothing_function=smoothie)
        valid_bleu = bleu_score * 100
        print(f"Validation BLEU: {valid_bleu:.2f}%")
        return valid_bleu



from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluate(dictionary):
    model.eval()  # Set the model to evaluation mode
    references = []  # Ground truth sentences
    hypotheses = []  # Predicted sentences

    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader_test, desc="Evaluation"):
            # Forward pass
            output__ = model(batch_x.to(device), batch_x.to(device))
            output = first_sentence = model.beam_search_decoding(output__)  # 第一句话
            for i in range(3):
                output_ = model(first_sentence.to(device), first_sentence.to(device))
                next_sentence = model.beam_search_decoding(output_)
                output = torch.cat((output, next_sentence), dim=1)
                first_sentence = next_sentence

            # Convert indices to words
            reference = [[dictionary.tkn2word[idx] for idx in sent if idx != dictionary.word2tkn['[PAD]']] for sent in batch_y.tolist()]

            references.extend(reference)

            # 外部循环迭代每个样本
            for sample_output in output:
                hypothesis = []  # 存储当前样本的假设序列
                # 内部循环迭代每个时间步
                i = 0
                for idx in sample_output.squeeze().tolist():
                    if idx == dataset.dictionary.word2tkn['[BOS]'] and i != 0:
                        i += 1
                        continue  # 跳过BOS标记
                    if idx != dataset.dictionary.word2tkn['[PAD]']:
                        word = dataset.dictionary.tkn2word[idx]
                        hypothesis.append(word)
                # 将当前样本的假设序列添加到总列表中
                hypotheses.append(hypothesis)

    # Calculate BLEU score
    smoothie = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, weights=(0.5, 0.5), smoothing_function=smoothie)
    print(f"BLEU Score: {bleu_score * 100:.2f}%")




if __name__ == '__main__':
    dataset_folder = './story_genaration_dataset'
    output_folder = './output'

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # -----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim = 300  # 每个词向量的维度
    max_token_per_sent = 50  # 每个句子预设的最大 token 数
    batch_size = 16
    num_epochs = 5
    lr = 1e-4
    # ------------------------------------------------------end------------------------------------------------------#

    dataset = Corpus(dataset_folder, max_token_per_sent)

    # print(dataset.train.tensors)

    vocab_size = len(dataset.dictionary.tkn2word)

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    max_seq_length = 50
    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, device, dataset.dictionary).to(device)

    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # 进行训练
    train()

    # 对测试集进行预测
    evaluate(dataset.dictionary)