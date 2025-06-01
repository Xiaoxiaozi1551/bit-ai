import os
import json

import gensim
import jieba
import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

class Dictionary(object):
    def __init__(self, path):

        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]

        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]

''' 任务1、2的Corpus '''
class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。

    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)

        # print(self.train.tensors)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置
        # 加载预训练的嵌入模型
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
            'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', binary=False)
        embedding_dim = embedding_model.vector_size

        # 创建嵌入权重矩阵
        vocab_size = len(self.dictionary.tkn2word)
        self.embedding_weight = np.zeros((vocab_size, embedding_dim))

        # 初始化[PAD]和[UNK]标记的嵌入权重
        self.embedding_weight[0] = np.zeros(embedding_dim)  # [PAD]
        # self.embedding_weight[1] = np.random.uniform(-0.25, 0.25, embedding_dim)  # [UNK]

        # 使用预训练的嵌入填充嵌入权重矩阵
        for word, idx in self.dictionary.word2tkn.items():
            if word in embedding_model:
                self.embedding_weight[idx] = embedding_model[word]
            else:
                self.embedding_weight[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)  # [UNK]
        self.embedding_weight = torch.from_numpy(self.embedding_weight).float()

        #------------------------------------------------------end------------------------------------------------------#

    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                #-----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词

                # 使用jieba进行中文分词
                sent = list(jieba.cut(sent, cut_all=True))

                #------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in sent:
                    self.dictionary.add_word(word)

                ids = []
                for word in sent:
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad(ids))

                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()

        return TensorDataset(idss, labels)

''' 任务3的Corpus '''


# class Corpus(object):
#     '''
#     完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
#
#     该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
#     '''
#
#     def __init__(self, path, max_token_per_sent):
#         self.dictionary = Dictionary(path)
#
#         self.max_token_per_sent = max_token_per_sent
#
#         self.tokenizer = BertTokenizer.from_pretrained('model/bert-base-chinese/')
#         self.train = self.tokenize(os.path.join(path, 'train.json'))
#         self.valid = self.tokenize(os.path.join(path, 'dev.json'))
#         self.test = self.tokenize(os.path.join(path, 'test.json'), True)
#
#     def tokenize(self, path, test_mode=False):
#         '''
#         处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
#         '''
#         idss = []
#         attentionn = []
#         totype = []
#         labels = []
#         with open(path, 'r', encoding='utf8') as f:
#             for line in f:
#                 one_data = json.loads(line)  # 读取一条数据
#                 sent = one_data['sentence']
#                 # 使用BertTokenizer对句子进行分词和编码
#                 encoded_input = self.tokenizer.encode_plus(
#                     sent,
#                     add_special_tokens=True,
#                     max_length=self.max_token_per_sent,
#                     padding='max_length',
#                     truncation=True,
#                     return_tensors='pt')
#
#
#                 input_ids = encoded_input['input_ids'].squeeze(0)
#                 attention_mask = encoded_input['attention_mask'].squeeze(0)
#                 token_type_ids = encoded_input['token_type_ids'].squeeze(0)
#                 idss.append(input_ids)
#                 attentionn.append(attention_mask)
#                 totype.append(token_type_ids)
#
#                 # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
#                 if test_mode:
#                     label = json.loads(line)['id']
#                     labels.append(label)
#                 else:
#                     label = json.loads(line)['label']
#                     labels.append(self.dictionary.label2idx[label])
#
#             idss = torch.tensor(np.array(idss))
#             attentionn = torch.tensor(np.array(attentionn))
#             totype = torch.tensor((np.array(totype)))
#             labels = torch.tensor(np.array(labels)).long()
#             # print(labels)
#             # print(idss.size())
#             # print(attentionn.size())
#             # print(labels.size())
#
#         return TensorDataset(idss, attentionn, totype, labels)