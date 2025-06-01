import json


class Dictionary(object):
    def __init__(self):
        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]

vocab_dict = Dictionary()

lines = open('data/data_train.txt', 'r', encoding='utf-8').readlines()
for line in lines:
    line = line.split('\t')
    text = line[3][:-1]
    for str_ in text:
        vocab_dict.add_word(str_)

lines = open('data/data_val.txt', 'r', encoding='utf-8').readlines()
for line in lines:
    line = line.split('\t')
    text = line[3][:-1]
    for str_ in text:
        vocab_dict.add_word(str_)

print(vocab_dict.word2tkn)

with open('word2tkn.json', 'w', encoding='utf-8') as json_file:
    json.dump(vocab_dict.word2tkn, json_file, ensure_ascii=False)