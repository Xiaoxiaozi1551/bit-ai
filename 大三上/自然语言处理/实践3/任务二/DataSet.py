import csv
import os
import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from transformers import GPT2Tokenizer

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
        self.tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2', padding_side='left')
        self.train = self.tokenize(os.path.join(path, 'ROCStories_train.csv'))
        self.valid = self.tokenize_test(os.path.join(path, 'ROCStories_val.csv'))
        self.test = self.tokenize_test(os.path.join(path, 'ROCStories_test.csv'), True)

    def tokenize(self, path, test_mode=False):
        idss = []
        stories = []
        attentionn = []
        totype = []

        # 使用 GPT-2 分词器
        i = 0
        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # i += 1
                # if i == 500:
                #     break
                story = {
                    'storyid': row['storyid'],
                    'storytitle': row['storytitle'],
                    'sentences': [row['sentence1'], row['sentence2'], row['sentence3'], row['sentence4'],
                                  row['sentence5']]
                }
                sent = ''.join(story['sentences'])
                self.tokenizer.add_special_tokens({'pad_token': 'pad'})
                encoded_input = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=60,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                input_ids = encoded_input['input_ids']
                # input_ids = input_ids.squeeze(0)

                attention_mask = encoded_input['attention_mask']
                # token_type_ids = encoded_input['token_type_ids']
                idss.append(input_ids)
                attentionn.append(attention_mask)
                # totype.append(token_type_ids)

                stories.append(story)
            idss = pad_sequence(idss, batch_first=True, padding_value=0)
            attentionn = pad_sequence(attentionn, batch_first=True, padding_value=0)
            # totype = torch.tensor(np.array(totype))\
            idss = torch.tensor(np.array(idss))
            attentionn = torch.tensor(np.array(attentionn))

        return TensorDataset(idss, attentionn)

    def tokenize_test(self, path, test_mode=False):
        idss = []
        stories = []
        attentionn = []
        idss_target = []
        attentionn_target = []
        totype = []

        # 使用 GPT-2 分词器
        i = 0
        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # i += 1
                # if i == 500:
                #     break
                story = {
                    'storyid': row['storyid'],
                    'storytitle': row['storytitle'],
                    'sentences': [row['sentence1'], row['sentence2'], row['sentence3'], row['sentence4'],
                                  row['sentence5']]
                }
                # 第一句
                sent = ''.join(story['sentences'][0])
                self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
                # encoded_input = self.tokenizer.encode_plus(
                #     sent,
                #     add_special_tokens=True,
                #     truncation=True,
                #     return_tensors='pt',
                # )
                encoded_input = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=60,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                # 定义最大长度和填充值
                max_length = 60
                padding_value = self.tokenizer.pad_token_id

                # # 进行左填充
                input_ids = encoded_input['input_ids']
                attention_mask = encoded_input['attention_mask']

                # 后四句
                sent = ''.join(story['sentences'][1:])
                encoded_input_target = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=60,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                input_ids_target = encoded_input_target['input_ids']
                attention_mask_target = encoded_input_target['attention_mask']

                idss.append(input_ids)
                attentionn.append(attention_mask)
                idss_target.append(input_ids_target)
                attentionn_target.append(attention_mask_target)

                stories.append(story)


            idss = pad_sequence(idss, batch_first=True, padding_value=0)
            attentionn = pad_sequence(attentionn, batch_first=True, padding_value=0)
            idss_target = pad_sequence(idss_target, batch_first=True, padding_value=0)
            attentionn_target = pad_sequence(attentionn_target, batch_first=True, padding_value=0)
            # totype = torch.tensor(np.array(totype))
            idss = torch.tensor(np.array(idss))
            attentionn = torch.tensor(np.array(attentionn))
            idss_target = torch.tensor(np.array(idss_target))
            attentionn_target = torch.tensor(np.array(attentionn_target))

        return TensorDataset(idss, attentionn, idss_target, attentionn_target)
