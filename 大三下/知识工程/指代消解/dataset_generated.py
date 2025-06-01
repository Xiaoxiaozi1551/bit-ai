import json
import os
import random

from datasets import Dataset


task = []
class DataSet(Dataset):
    def __init__(self, img_dir, label_file, task2vector, label2vector):
        # print(task2vector)
        # 添加数据集的初始化内容
        self.img_dir = img_dir
        self.task2vector = task2vector
        self.label2vector = label2vector
        self.word_vector, self.label = self.load_samples(label_file)
        self.X, self.y = self.data_and_label()
        # self.label = self.load_samples(label_file)

    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        for i in range(len(self.word_vector)):
            for j in range(len(self.word_vector[i])):
                identity = self.word_vector[i][j]
                label = self.label2vector[i][j]
        return identity, label

    def __len__(self):
        # 添加len函数的相关内
        sum_ = 0
        for i in range(len(self.word_vector)):
            sum_ += len(self.word_vector[i])
        return sum_

    def load_samples(self, label_file):
        label_vector = []

        for file_name in os.listdir(label_file):
            with open(label_file + file_name, 'r', encoding='GBK') as fp:
                data = json.load(fp)
                if data is not None:
                    task_num = data['0']['id']
                    word_index = data['pronoun']['indexFront']  # 代词的位置
                    startIndex = data['0']['indexFront']
                    endIndex = data['0']['indexBehind']

                    if task_num != '19980127-09-001-033' and word_index < len(self.task2vector[task_num]):
                        if task_num not in task:
                            task.append(task_num)
                            # print(task_num, len(self.task2vector[task_num][word_index]))
                            self.task2vector[task_num][word_index].extend(self.task2vector[task_num][word_index])
                        self.task2vector[task_num][word_index][13] = word_index
                        leng = len(self.label2vector[task_num])
                        # print(leng, endIndex+1)
                        for i in range(startIndex, min(endIndex + 1, leng)):
                            # print(self.label2vector[task_num])
                            self.label2vector[task_num][i] = 1

                        if data['antecedentNum'] == 2:
                            startIndex = data['1']['indexFront']
                            endIndex = data['1']['indexBehind']
                            for i in range(startIndex, min(endIndex + 1, leng)):
                                    # print(self.label2vector[task_num])
                                    self.label2vector[task_num][i] = 1

        word_vector = []
        for item in self.task2vector.items():
            if item[0] in task:
                word_vector.append(item[1])
        for item in self.label2vector.items():
            if item[0] in task:
                label_vector.append(item[1])
        # for item in self.label2vector.values():
        #     label_vector.append(item)
        for item_list in word_vector:
            location = item_list[-1][13]  # 初始化
            if len(item_list[-1]) > 14:
                location_list = [item for item in item_list[-1][14:]]
            else:
                location_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                item_list[-1].extend(location_list)

            for i in range(len(item_list) - 2, -1, -1):
                if len(item_list[i]) == 28:
                    location = item_list[i][13]
                    location_list = [item for item in item_list[i][14:]]
                    # print(location_list)
                    item_list[i][13] = i - location
                else:
                    # print(1)
                    item_list[i][13] = i - location
                    item_list[i].extend(location_list)
        # print(word_vector)
        # for item_list in word_vector:
        #     for item in item_list:
        #         if len(item) != 28:
        #             print(11111)

        return word_vector, label_vector

    def data_and_label(self):
        data_ = []
        label_ = []
        data_X = []
        data_y = []
        for item_sentence in self.word_vector:
            for item_word in item_sentence:
                data_.append(item_word)
        for item_sentence in self.label:
            for item_word in item_sentence:
                label_.append(item_word)
        sum_pos = 0
        sum_neg = 0
        for i in range(len(data_)):
            if label_[i] == 1:
                data_X.append(data_[i])
                data_y.append(label_[i])
                sum_pos += 1
            else:
                sum_neg += 1
        print("sum_pos:", sum_pos)
        print("sum_neg:", sum_neg)
        for i in range(int(sum_pos//2)):
            num = random.randint(0, len(data_)-1)
            data_X.append(data_[num])
            data_y.append(label_[num])

        c = list(zip(data_X, data_y))
        random.shuffle(c)
        data_X, data_y = zip(*c)
        # for i in range(len(data_X)):
        #     print("data:", data_X[i], "label:", data_y[i])
        return data_X, data_y


# 构造词特征
def word_feature():
    # 加载停用词
    stopwords_path = './cn_stopwords.txt'
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
        file.close()

    # 词的特征构建
    text_data = []
    path_word = "./1998-01-2003版-带音.txt"
    with open(path_word, 'r') as file:
        for line in file:
            if line != "\n":  # 把空行去掉
                data_line = line.strip("\n").split()  # 对读入的去除\n并按空格分词
                text_data.append(data_line)
        file.close()

    word_data = []  # 去掉词性的词
    wordType_data = []  # 词性列表

    for item_list in text_data:
        temp_word = []
        temp_wordType = []
        for item in item_list:
            temp = item.split('/')
            temp_word.append(temp[0])
            temp_wordType.append('/' + temp[1])

        word_data.append(temp_word)
        wordType_data.append(temp_wordType)

    # print(word_data)

    # 去停用词和统计词频    先不用
    # wordcount = {}
    # for item_list in word_data:
    #     for item in item_list:
    #         if item not in stopwords:
    #             wordcount[item] = wordcount.get(item, 0) + 1
    # print(sorted(wordcount.items(), key=lambda x: x[1], reverse=True)[:10])

    # 词性特征
    wordType_other = {'Noun': 1,
                'Pronoun': 2,
                'Verb': 3,
                'Time': 4,
                'Numeral': 5,
                'Measure': 6,
                'Adverb': 7,
                'Auxiliary': 8,
                'Punctuation': 9,
                'Neither': 10}
    wordType = {'Noun': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Pronoun': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Verb': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                'Time': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                'Numeral': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'Measure': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'Adverb': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                'Auxiliary': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'Punctuation': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                'Neither': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # 是否为姓名+代词位置
    Name_sur = {'surname': [1, 0, 0],
                'name': [0, 1, 0],
                'neither': [0, 0, 0]}

    word_vector = []  # 把文本全部转化为特征向量
    task2vector = {}  # 任务标号：对应向量
    label2vector = {}  # 标签初始化
    for i in range(len(word_data)):
        temp_sentence = []
        temp_label = []
        for j in range(len(word_data[i])):
            temp = []
            if j != 0:  # 去掉任务编号
                temp_label.append(0)
                # print(wordType_data[i][j][1])
                if wordType_data[i][j][1] == 'n' or wordType_data[i][j][1] == 'N':
                    temp.extend(wordType['Noun'])
                elif wordType_data[i][j][1] == 'r' or wordType_data[i][j][1] == 'R':
                    temp.extend(wordType['Pronoun'])
                elif wordType_data[i][j][1] == 'v' or wordType_data[i][j][1] == 'V':
                    temp.extend(wordType['Verb'])
                elif wordType_data[i][j][1] == 't' or wordType_data[i][j][1] == 'T':
                    temp.extend(wordType['Time'])
                elif wordType_data[i][j][1] == 'm' or wordType_data[i][j][1] == 'M':
                    temp.extend(wordType['Numeral'])
                elif wordType_data[i][j][1] == 'q' or wordType_data[i][j][1] == 'Q':
                    temp.extend(wordType['Measure'])
                elif wordType_data[i][j][1] == 'd' or wordType_data[i][j][1] == 'D':
                    temp.extend(wordType['Adverb'])
                elif wordType_data[i][j][1] == 'u' or wordType_data[i][j][1] == 'U':
                    temp.extend(wordType['Auxiliary'])
                elif wordType_data[i][j][1] == 'w' or wordType_data[i][j][1] == 'W':
                    temp.extend(wordType['Punctuation'])
                else:
                    temp.extend(wordType['Neither'])


                for index in range(9, 11):
                    if index == 9:
                        num = -1
                    else:
                        num = 1
                    if j+num >= len(word_data[i]) or j+num <= 1:
                        continue
                    else:
                        word_Types = wordType_data[i][j+num][1]

                    if word_Types == 'n' or word_Types == 'N':
                        temp[index] = wordType_other['Noun']
                    elif word_Types == 'r' or word_Types == 'R':
                        temp[index] = wordType_other['Pronoun']
                    elif word_Types == 'v' or word_Types == 'V':
                        temp[index] = wordType_other['Verb']
                    elif word_Types == 't' or word_Types == 'T':
                        temp[index] = wordType_other['Time']
                    elif word_Types == 'm' or word_Types == 'M':
                        temp[index] = wordType_other['Numeral']
                    elif word_Types == 'q' or word_Types == 'Q':
                        temp[index] = wordType_other['Measure']
                    elif word_Types == 'd' or word_Types == 'D':
                        temp[index] = wordType_other['Adverb']
                    elif word_Types == 'u' or word_Types == 'U':
                        temp[index] = wordType_other['Auxiliary']
                    elif word_Types == 'w' or word_Types == 'W':
                        temp[index] = wordType_other['Punctuation']
                    else:
                        temp[index] = wordType_other['Neither']

                if wordType_data[i][j] == '/nrf':
                    temp.extend(Name_sur['surname'])
                elif wordType_data[i][j] == '/nrg':
                    temp.extend(Name_sur['name'])
                else:
                    temp.extend(Name_sur['neither'])
                temp_sentence.append(temp)
            else:
                task_number = word_data[i][j]
        task2vector[task_number] = temp_sentence
        label2vector[task_number] = temp_label
        word_vector.append(temp_sentence)

    # print(task2vector)
    return task2vector, label2vector