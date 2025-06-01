import csv
import os
import json

import gensim
import numpy as np
import torch
from nltk import word_tokenize
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

class Dictionary(object):
    def __init__(self, path):

        self.word2tkn = {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3}  # 初始词汇表
        self.tkn2word = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, 'ROCStories_train.csv'))
        self.valid = self.tokenize(os.path.join(path, 'ROCStories_val.csv'))
        self.test = self.tokenize(os.path.join(path, 'ROCStories_test.csv'), True)

    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def pad_left(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))] + origin_token_seq

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        targett = []

        stories = []

        vocab_size = len(self.dictionary.word2tkn)
        i = 0
        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                story = {
                    'storyid': row['storyid'],
                    'storytitle': row['storytitle'],
                    'sentences': [row['sentence1'], row['sentence2'], row['sentence3'], row['sentence4'],
                                  row['sentence5']]
                }

                # 分词
                tokenized_sentences = []
                for sentence in story['sentences']:
                    tokens = word_tokenize(sentence)
                    tokens.insert(0, '[BOS]')
                    tokens.append('[EOS]')
                    tokenized_sentences.append(tokens)

                story['tokenized_sentences'] = tokenized_sentences
                stories.append(story)

                # 向词典中添加词
                for sent in story['tokenized_sentences']:
                    for word in sent:
                        self.dictionary.add_word(word)

                ids = []
                for word in story['tokenized_sentences'][0]:
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad_left(ids))


                target = []
                for sent in story['tokenized_sentences'][1:]:
                    for word in sent:
                        target.append(self.dictionary.word2tkn[word])
                targett.append(self.pad(target))

            idss = torch.tensor(np.array(idss))
            targett = torch.tensor(np.array(targett)).long()

        return TensorDataset(idss, targett)