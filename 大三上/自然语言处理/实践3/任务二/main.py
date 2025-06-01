import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataSet import Corpus
# from My_Model import TransformerModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

'''
无微调
'''
# def train():
#     max_valid_bleu = 0
#     accumulation_steps = 4  # 4 次小批次后更新一次梯度
#
#     for epoch in range(num_epochs):
#         model.train()  # Set the model to training mode
#         train_loss = 0.0
#
#         # Training loop
#         for batch_idx, (batch_x, attention) in enumerate(tqdm(data_loader_train, desc=f"Epoch {epoch} Training")):
#             optimizer.zero_grad()  # Zero the gradients
#
#             batch_x = batch_x.to(device)
#             attention = attention.to(device)
#
#             # Forward pass
#             output = model(batch_x, labels=batch_x, attention_mask=attention)
#             loss = output.loss
#
#             # Backward pass
#             loss.backward()
#
#             train_loss += loss.item()
#
#         # Validation
#         valid_bleu = valid()
#
#         if valid_bleu > max_valid_bleu:
#             # Save the model when validation BLEU improves
#             torch.save(model.state_dict(), os.path.join(output_folder, "model_test.ckpt"))
#             max_valid_bleu = valid_bleu
#
#         print(f"epoch: {epoch}, train loss: {train_loss:.4f}, "
#               f"valid BLEU: {valid_bleu * 100:.2f}%")


'''
部分解冻
'''
# def train():
#     max_valid_bleu = 0
#     accumulation_steps = 4  # 4 次小批次后更新一次梯度
#     unfreeze_level = -1  # 初始解冻层级（例如，GPT-2模型中的所有层）
#
#     for epoch in range(num_epochs):
#         model.train()  # 设置模型为训练模式
#         train_loss = 0.0
#
#         # 解冻更低层级的参数
#         if unfreeze_level >= 0:
#             for param in model.transformer.h[unfreeze_level:].parameters():
#                 param.requires_grad = True
#
#         # Training loop
#         for batch_idx, (batch_x, attention) in enumerate(tqdm(data_loader_train, desc=f"Epoch {epoch} Training")):
#             optimizer.zero_grad()  # 梯度置零
#
#             batch_x = batch_x.to(device)
#             attention = attention.to(device)
#
#             # Forward pass
#             output = model(batch_x, labels=batch_x, attention_mask=attention)
#             loss = output.loss
#
#             # Backward pass
#             loss.backward()
#
#             # Accumulate gradients
#             # if (batch_idx + 1) % accumulation_steps == 0:
#             #     optimizer.step()
#             #     optimizer.zero_grad()  # 更新后将梯度置零
#
#             train_loss += loss.item()
#
#         # Validation
#         valid_bleu = valid()
#
#         if valid_bleu > max_valid_bleu:
#             # 当验证BLEU提高时保存模型
#             torch.save(model.state_dict(), os.path.join(output_folder, "model_test_3.ckpt"))
#             max_valid_bleu = valid_bleu
#
#         print(f"epoch: {epoch}, train loss: {train_loss:.4f}, "
#               f"valid BLEU: {valid_bleu * 100:.2f}%")
#
#         # 更新解冻层级的条件
#         if epoch == 4:  # 示例条件：在第4个训练周期后解冻更低层级的参数
#             unfreeze_level -= 1  # 解冻更低层级的参数

'''梯度累计'''
def train():
    max_valid_bleu = 0
    accumulation_steps = 4  # 4 次小批次后更新一次梯度
    total_loss = 0  # 用于累积每个小批次的损失

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0

        # Training loop
        for batch_idx, (batch_x, attention) in enumerate(tqdm(data_loader_train, desc=f"Epoch {epoch} Training")):
            optimizer.zero_grad()  # 清零梯度

            batch_x = batch_x.to(device)
            attention = attention.to(device)

            # Forward pass
            output = model(batch_x, labels=batch_x, attention_mask=attention)
            loss = output.loss

            # Backward pass
            loss.backward()

            # 累积梯度
            total_loss += loss.item()

            if (batch_idx + 1) % accumulation_steps == 0:
                # 更新梯度
                optimizer.step()
                optimizer.zero_grad()  # 清零梯度

                train_loss += total_loss / accumulation_steps
                total_loss = 0

        # Validation
        valid_bleu = valid()

        if valid_bleu > max_valid_bleu:
            # 当验证BLEU提升时保存模型
            torch.save(model.state_dict(), os.path.join(output_folder, "model_test_4.ckpt"))
            max_valid_bleu = valid_bleu

        print(f"epoch: {epoch}, train loss: {train_loss:.4f}, "
              f"valid BLEU: {valid_bleu * 100:.2f}%")


def valid():
    model.eval()  # 将模型设置为评估模式
    references = []  # 真实的句子列表
    hypotheses = []  # 预测的句子列表

    with torch.no_grad():
        for batch_x, attention, batch_x_target, attention_target in tqdm(data_loader_valid, desc="Validation"):
            batch_x = batch_x.to(device)
            attention = attention.to(device)
            batch_x = batch_x.squeeze(dim=1)
            attention = attention.squeeze(dim=1)
            # print('batch', batch_x)
            output = model.generate(input_ids=batch_x,
                                    attention_mask=attention,
                                    max_length=110,
                                    pad_token_id=dataset.tokenizer.pad_token_id,
                                    temperature=0.5,
                                    num_return_sequences=1)
            # print('output', output)
            # 解码生成的文本

            for i in range(output.size(0)):
                generated_text = dataset.tokenizer.decode(output[i][0], skip_special_tokens=True)
                hypotheses.append(generated_text)

            # 将真实句子和预测句子添加到列表中
            for i in range(batch_x_target.size(0)):
                sent = dataset.tokenizer.decode(batch_x[i][0], skip_special_tokens=True) +\
                       dataset.tokenizer.decode(batch_x_target[i][0], skip_special_tokens=True)
                references.append(sent)


    # 计算BLEU分数
    smoothie = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, weights=(0.8, 0.2), smoothing_function=smoothie)

    return bleu_score


def evaluate(dictionary):
    model.eval()  # Set the model to evaluation mode
    references = []  # Ground truth sentences
    hypotheses = []  # Predicted sentences

    with torch.no_grad():
        for batch_x, attention, batch_x_target, attention_target in tqdm(data_loader_test, desc="Evaluation"):
            batch_x = batch_x.to(device)
            attention = attention.to(device)
            batch_x = batch_x.squeeze(dim=1)
            attention = attention.squeeze(dim=1)
            output = model.generate(input_ids=batch_x,
                                    attention_mask=attention,
                                    max_length=110,
                                    pad_token_id=dataset.tokenizer.pad_token_id,
                                    temperature=0.5,
                                    num_return_sequences=1)
            # 解码生成的文本

            for i in range(output.size(0)):
                generated_text = dataset.tokenizer.decode(output[i], skip_special_tokens=True)
                hypotheses.append(generated_text)
            # 将真实句子和预测句子添加到列表中
            for i in range(batch_x_target.size(0)):
                references.append(dataset.tokenizer.decode(batch_x_target[i][0], skip_special_tokens=True))

            '''测试'''
            # prompt = dataset.tokenizer.encode_plus('Hello world!', return_tensors='pt').to(device)
            # output = model.generate(input_ids=prompt['input_ids'],
            #                         attention_mask=prompt['attention_mask'],
            #                         max_length=100,
            #                         pad_token_id=dataset.tokenizer.pad_token_id,
            #                         no_repeat_ngram_size=2,
            #                         top_k=50,
            #                         top_p=0.7,
            #                         temperature=0.7,
            #                         num_return_sequences=1)
            # print(output)
            # print(dataset.tokenizer.decode(output[0], skip_special_tokens=True))
            '''ceshi'''

    # Calculate BLEU score
    smoothie = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, weights=(0.8, 0.2), smoothing_function=smoothie)
    print(f"BLEU Score: {bleu_score * 100:.2f}%")


if __name__ == '__main__':
    dataset_folder = './story_genaration_dataset'
    output_folder = './output'

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # -----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim = 300  # 每个词向量的维度
    max_token_per_sent = 60  # 每个句子预设的最大 token 数
    batch_size = 8
    num_epochs = 5
    lr = 1e-4
    # ------------------------------------------------------end------------------------------------------------------#

    dataset = Corpus(dataset_folder, max_token_per_sent)

    # print(dataset.train.tensors)

    vocab_size = len(dataset.dictionary.tkn2word)

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    # d_model = 512
    # nhead = 8
    # num_encoder_layers = 6
    # num_decoder_layers = 6
    # dim_feedforward = 2048
    # max_seq_length = 50
    # model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, device, dataset.dictionary).to(device)

    model = GPT2LMHeadModel.from_pretrained('model/gpt2').to(device)

    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # 进行训练
    train()

    # 对测试集进行预测
    evaluate(dataset.dictionary)