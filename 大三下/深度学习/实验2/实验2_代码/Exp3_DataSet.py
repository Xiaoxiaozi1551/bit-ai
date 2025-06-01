import json
import math
import re
from collections import Counter

import jieba
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from Exp3_Config import Training_Config


class Dictionary(object):
    def __init__(self):
        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


# 训练集和验证集
class TextDataSet(Dataset):
    """
    filepath：包含文本数据的文件路径。这是初始化TextDataSet类实例时提供的输入参数。
    self.original_data：一个列表，存储从文件中读取的原始数据。列表中的每个项表示一个数据条目，包含头实体、尾实体、关系和文本等信息。
    self.processed_data：一个列表，存储处理过的数据。列表中的每个项表示一个数据条目，其中文本使用字符到整数的映射转换为数值表示。
    self.head_pos_lists：一个列表，存储每个数据条目的头实体的相对位置编码。
    self.tail_pos_lists：一个列表，存储每个数据条目的尾实体的相对位置编码。
    self.labels：一个列表，存储每个数据条目的标签（关系ID）。
    self.config：一个Training_Config类的实例。它可能包含用于训练的配置设置。
    self.rel2id_table：一个字典，将关系名称映射到对应的ID。它从"rel2id.json"文件中加载。
    self.word_vocab：一个字典，将单词映射到它们的整数表示。它从"word2tkn.json"文件中加载。
    """
    def __init__(self, filepath):
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        self.processed_data = []
        # self.word_vocab = Dictionary()
        self.head_pos_lists = []
        self.tail_pos_lists = []
        self.labels = []
        self.config = Training_Config()
        with open("data/rel2id.json", 'r', encoding='utf-8') as f:
            self.rel2id_table = json.load(f)
        # 建立字符表
        with open('word2tkn.json', 'r', encoding='utf-8') as f:
            self.word_vocab = json.load(f)

        for line in lines:
            tmp = {}
            line = line.split('\t')
            #  去除占位符
            if line[0] == 'subject_placeholder':
                continue
            if line[1] == 'object_placeholder':
                continue
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['relation'] = line[2]
            tmp['text'] = line[3][:-1]
            self.original_data.append(tmp)
        # index_ = -1
        # for item in self.original_data:
        #     index_ += 1
        #     entity1 = item['head']
        #     entity2 = item['tail']
        #     temp_text = item['text']  # 读入句子
        #     if entity1.find(entity2) == -1 and entity2.find(entity1) == -1:
        #         temp_text = temp_text.replace(entity1, 'HeadEntity')
        #         temp_text = temp_text.replace(entity2, 'TailEntity')
        #     else:
        #         if entity1.find(entity2) != -1:
        #             temp_text = temp_text.replace(entity1, 'HeadEntity')
        #             temp_text = temp_text.replace(entity2, 'TailEntity')
        #         else:
        #             temp_text = temp_text.replace(entity2, 'TailEntity')
        #             temp_text = temp_text.replace(entity1, 'HeadEntity')
        #     segment_text = jieba.lcut(temp_text)
        #     result = self.transfer(segment_text, entity1, entity2)
        #     # 建立词表
        #     for word in result:
        #         self.word_vocab.add_word(word)
        #     # 获得位置向量
        #     head_pos = result.index(entity1)
        #     tail_pos = result.index(entity2)
        #     head_pos_list, tail_pos_list = self.relative_position_encoding(head_pos, tail_pos, len(result))
            tmp['text'] = tmp['text'].replace('#', '')
            head_pos = tmp['text'].find(tmp['head'])
            tail_pos = tmp['text'].find(tmp['tail'])
            head_pos_list, tail_pos_list = self.relative_position_encoding(head_pos, tail_pos, len(tmp['text']))  # 对当前句子进行相对位置编码
            head_pos_list = self.padding(head_pos_list, head_pos)  # 截断与填充
            tail_pos_list = self.padding(tail_pos_list, head_pos)

            self.head_pos_lists.append(head_pos_list)
            self.tail_pos_lists.append(tail_pos_list)
            self.labels.append(self.rel2id_table[1][tmp['relation']])

            # 根据字符表,将字符转为数字
            result = self.ch2id(tmp['text'])
            result = self.padding(result, head_pos)

            self.processed_data.append(result)

        label_counts = Counter(self.labels)
        for label, count in label_counts.items():
            print(f"Class {label}: {count} samples")
        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Samples in Each Class')
        plt.xticks(rotation=45)
        plt.show()

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.processed_data[index])
        pos = self.original_data[index]['text'].find(self.original_data[index]['head'])
        entity1 = self.ch2id(self.original_data[index]['head'])
        entity2 = self.ch2id(self.original_data[index]['tail'])
        entity1 = self.padding(entity1, pos)
        entity2 = self.padding(entity2, pos)
        HeadEntity = torch.LongTensor(entity1)
        TailEntity = torch.LongTensor(entity2)
        head_pos = torch.LongTensor(self.head_pos_lists[index])
        tail_pos = torch.LongTensor(self.tail_pos_lists[index])
        label = self.labels[index]
        # print(sentence.size)
        # print(sentence.shape, HeadEntity.shape, TailEntity.shape, head_pos.shape, tail_pos.shape)

        return sentence, HeadEntity, TailEntity, head_pos, tail_pos, label

    def __len__(self):
        return len(self.original_data)

    def ch2id(self, sentence):
        ch2id_list = []
        for ch in sentence:
            if ch in self.word_vocab.keys():
                ch2id_list.append(self.word_vocab[ch])
            else:
                ch2id_list.append(0)
        return ch2id_list

    def relative_position_encoding(self, entity1_pos, entity2_pos, sentence_length):
        # 计算头实体与尾实体之间的相对位置
        relative_position1 = [abs(i - entity1_pos) for i in range(sentence_length)]
        relative_position2 = [abs(i - entity2_pos) for i in range(sentence_length)]

        # 使用正弦函数和余弦函数生成位置编码
        # sinusoid_table = [[math.sin(pos / 1000 ** (i / sentence_length)) if i % 2 == 0 else math.cos(
        #     pos / 1000 ** ((i - 1) / sentence_length)) for i in range(sentence_length)] for pos in
        #                   range(sentence_length)]
        #
        # # 获取头实体和尾实体的相对位置编码
        # relative_position_encoding1 = [sinusoid_table[pos][rel_pos] for pos, rel_pos in enumerate(relative_position1)]
        # relative_position_encoding2 = [sinusoid_table[pos][rel_pos] for pos, rel_pos in enumerate(relative_position2)]

        # 返回相对位置编码
        return relative_position1, relative_position2

    def padding(self, result, head_pos):
        # 进行填充
        if len(result) >= self.config.max_sentence_length:
            if head_pos + self.config.max_sentence_length <= len(result):
                result = result[head_pos:head_pos + self.config.max_sentence_length]
            else:
                result = result[len(result) - self.config.max_sentence_length:]
        else:
            result = result + [0] * (self.config.max_sentence_length - len(result))
        return result


# 测试集是没有标签的，因此函数会略有不同
class TestDataSet(Dataset):
    """
    filepath：包含文本数据的文件路径。这是初始化TextDataSet类实例时提供的输入参数。
    self.original_data：一个列表，存储从文件中读取的原始数据。列表中的每个项表示一个数据条目，包含头实体、尾实体、关系和文本等信息。
    self.processed_data：一个列表，存储处理过的数据。列表中的每个项表示一个数据条目，其中文本使用字符到整数的映射转换为数值表示。
    self.head_pos_lists：一个列表，存储每个数据条目的头实体的相对位置编码。
    self.tail_pos_lists：一个列表，存储每个数据条目的尾实体的相对位置编码。
    self.labels：一个列表，存储每个数据条目的标签（关系ID）。
    self.config：一个Training_Config类的实例。它可能包含用于训练的配置设置。
    self.word_vocab：一个字典，将单词映射到它们的整数表示。它从"word2tkn.json"文件中加载。
    """
    def __init__(self, filepath):
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        self.processed_data = []
        self.head_pos_lists = []
        self.tail_pos_lists = []
        self.labels = []
        self.config = Training_Config()
        # 建立字符表
        with open('word2tkn.json', 'r', encoding='utf-8') as f:
            self.word_vocab = json.load(f)

        for line in lines:
            tmp = {}
            line = line.split('\t')
            #  去除占位符
            if line[0] == 'subject_placeholder':
                continue
            if line[1] == 'object_placeholder':
                continue
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['text'] = line[2][:-1]
            self.original_data.append(tmp)

            tmp['text'] = tmp['text'].replace('#', '')
            head_pos = tmp['text'].find(tmp['head'])
            tail_pos = tmp['text'].find(tmp['tail'])
            head_pos_list, tail_pos_list = self.relative_position_encoding(head_pos, tail_pos,
                                                                           len(tmp['text']))  # 对当前句子进行相对位置编码
            head_pos_list = self.padding(head_pos_list, head_pos)  # 截断与填充
            tail_pos_list = self.padding(tail_pos_list, head_pos)

            self.head_pos_lists.append(head_pos_list)
            self.tail_pos_lists.append(tail_pos_list)
            # 根据字符表,将字符转为数字
            result = self.ch2id(tmp['text'])
            result = self.padding(result, head_pos)

            self.processed_data.append(result)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.processed_data[index])
        pos = self.original_data[index]['text'].find(self.original_data[index]['head'])
        entity1 = self.ch2id(self.original_data[index]['head'])
        entity2 = self.ch2id(self.original_data[index]['tail'])
        entity1 = self.padding(entity1, pos)
        entity2 = self.padding(entity2, pos)
        HeadEntity = torch.LongTensor(entity1)
        TailEntity = torch.LongTensor(entity2)
        head_pos = torch.LongTensor(self.head_pos_lists[index])
        tail_pos = torch.LongTensor(self.tail_pos_lists[index])

        return sentence, HeadEntity, TailEntity, head_pos, tail_pos

    def __len__(self):
        return len(self.original_data)

    def ch2id(self, sentence):
        ch2id_list = []
        for ch in sentence:
            if ch in self.word_vocab.keys():
                ch2id_list.append(self.word_vocab[ch])
            else:
                ch2id_list.append(0)
        return ch2id_list

    def relative_position_encoding(self, entity1_pos, entity2_pos, sentence_length):
        # 计算头实体与尾实体之间的相对位置
        relative_position1 = [abs(i - entity1_pos) for i in range(sentence_length)]
        relative_position2 = [abs(i - entity2_pos) for i in range(sentence_length)]

        # 使用正弦函数和余弦函数生成位置编码
        # sinusoid_table = [[math.sin(pos / 1000 ** (i / sentence_length)) if i % 2 == 0 else math.cos(
        #     pos / 1000 ** ((i - 1) / sentence_length)) for i in range(sentence_length)] for pos in
        #                   range(sentence_length)]
        #
        # # 获取头实体和尾实体的相对位置编码
        # relative_position_encoding1 = [sinusoid_table[pos][rel_pos] for pos, rel_pos in enumerate(relative_position1)]
        # relative_position_encoding2 = [sinusoid_table[pos][rel_pos] for pos, rel_pos in enumerate(relative_position2)]

        # 返回相对位置编码
        return relative_position1, relative_position2

    def padding(self, result, head_pos):
        # 进行填充
        if len(result) >= self.config.max_sentence_length:
            if head_pos + self.config.max_sentence_length <= len(result):
                result = result[head_pos:head_pos + self.config.max_sentence_length]
            else:
                result = result[len(result) - self.config.max_sentence_length:]
        else:
            result = result + [0] * (self.config.max_sentence_length - len(result))
        return result


if __name__ == "__main__":
    trainset = TextDataSet(filepath="./data/data_train.txt")
    testset = TestDataSet(filepath="./data/test_exp3.txt")
    print(trainset.__getitem__(3))
    print("训练集长度为：", len(trainset))
    print("测试集长度为：", len(testset))
