import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN_Model(nn.Module):
    """
    configs: 包含模型配置的对象。这个参数用于初始化模型时传递各种配置选项。
    vocab_size: 词汇表的大小，表示词嵌入层的输入维度。
    embedding_dimension: 词嵌入的维度，表示词嵌入层的输出维度。
    label_num: 标签的数量，表示全连接层的输出维度。
    self.embed: 词嵌入层（Embedding Layer），将词汇表中的单词索引转换为密集向量表示。
    self.dropout: Dropout 层，用于在训练过程中随机丢弃部分神经元，以减少过拟合。
    self.conv1, self.conv2, self.conv3: 卷积层（Convolutional Layers），用于提取句子中的局部特征。这里使用了三个不同大小的卷积核（kernel size 为 3、4、5），每个卷积核的输入通道数为 embedding_dimension，输出通道数为 100。
    self.fc: 全连接层（Fully Connected Layer），将卷积层的输出映射到标签的数量。输入维度见 forward 处理，因为有三个卷积核，所以输入维度是 300。
    """
    def __init__(self, configs):
        super(TextCNN_Model, self).__init__()

        vocab_size = configs.vocab_size
        embedding_dimension = configs.embedding_dimension
        label_num = configs.label_num

        # 词嵌入和dropout
        self.embed = nn.Embedding(vocab_size, embedding_dimension)
        self.dropout = nn.Dropout(configs.dropout)

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=embedding_dimension, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embedding_dimension, out_channels=100, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embedding_dimension, out_channels=100, kernel_size=5)

        # 全连接层
        self.fc = nn.Linear(20300, label_num)  # 因为有3个卷积核，所以输入维度是300

    # def forward(self, sentence, head_entity, tail_entity, head_pos, tail_pos):
    #     # 嵌入文本部分
    #     embed_sentence = self.embed(sentence)  # [batch_size, max_sentence_length, embedding_dimension]
    #     embed_sentence = self.dropout(embed_sentence)
    #
    #     # 嵌入实体部分
    #     embed_head_entity = self.embed(head_entity)
    #     embed_tail_entity = self.embed(tail_entity)
    #
    #     # 将head_pos和tail_pos转换为嵌入向量
    #     embed_head_pos = self.embed(head_pos)
    #     embed_tail_pos = self.embed(tail_pos)
    #
    #     # # 将head_entity和tail_entity的嵌入向量与位置信息相加
    #     # combined_head = embed_head_entity + embed_head_pos
    #     # combined_tail = embed_tail_entity + embed_tail_pos
    #     # 将head_entity和tail_entity的嵌入向量与位置信息拼接
    #     combined_head = torch.cat((embed_head_entity, embed_head_pos), dim=1)
    #     combined_tail = torch.cat((embed_tail_entity, embed_tail_pos), dim=1)
    #
    #     # 执行卷积和池化操作
    #     x1 = F.relu(self.conv1(embed_sentence.transpose(1, 2)))  # [batch_size, out_channels, max_sentence_length - kernel_size + 1]
    #     x2 = F.relu(self.conv2(embed_sentence.transpose(1, 2)))
    #     x3 = F.relu(self.conv3(embed_sentence.transpose(1, 2)))
    #
    #     # 池化
    #     x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch_size, out_channels]
    #     x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
    #     x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
    #
    #     # 连接卷积层的输出
    #     x = torch.cat((x1, x2, x3), dim=1)  # [batch_size, 3*out_channels]
    #
    #     # 将实体部分的信息连接到卷积层的输出中
    #     # x = x.unsqueeze(1)
    #     combined_head = combined_head.view(combined_head.size(0), -1)  # 将combined_head从三维变为二维
    #     combined_tail = combined_tail.view(combined_tail.size(0), -1)  # 将combined_tail从三维变为二维
    #     # print(x.shape, combined_head.shape, combined_tail.shape)
    #     x = torch.cat((x, combined_head, combined_tail), dim=1)
    #
    #     # 全连接层
    #     x = self.fc(x)
    #
    #     return x

    def forward(self, sentence, head_entity, tail_entity, head_pos, tail_pos):
        # 嵌入文本部分
        embed_sentence = self.embed(sentence)  # [batch_size, max_sentence_length, embedding_dimension]
        # embed_sentence = self.dropout(embed_sentence)

        # 嵌入实体部分
        embed_head_entity = self.embed(head_entity)
        embed_tail_entity = self.embed(tail_entity)

        # 将head_pos和tail_pos转换为嵌入向量
        embed_head_pos = self.embed(head_pos)
        embed_tail_pos = self.embed(tail_pos)

        # # 将head_entity和tail_entity的嵌入向量与位置信息相加
        # combined_head = embed_head_entity + embed_head_pos
        # combined_tail = embed_tail_entity + embed_tail_pos
        combined_entity = torch.cat((embed_head_entity, embed_tail_entity), dim=1)
        combined_pos = torch.cat((embed_head_pos, embed_tail_pos), dim=1)

        # 执行卷积和池化操作
        embed_sentence = torch.cat((embed_sentence, combined_pos), dim=1)
        embed_sentence = self.dropout(embed_sentence)
        x1 = F.relu(self.conv1(embed_sentence.transpose(1, 2)))  # [batch_size, out_channels, max_sentence_length - kernel_size + 1]
        x2 = F.relu(self.conv2(embed_sentence.transpose(1, 2)))
        x3 = F.relu(self.conv3(embed_sentence.transpose(1, 2)))

        # 池化
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch_size, out_channels]
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        # 连接卷积层的输出
        x = torch.cat((x1, x2, x3), dim=1)  # [batch_size, 3*out_channels]

        combined_entity = combined_entity.view(combined_entity.size(0), -1)
        x = torch.cat((x, combined_entity), dim=1)

        # 全连接层
        x = self.fc(x)

        return x
