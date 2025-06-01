import torch.nn as nn
import torch as torch
import math
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.size())
        # print(self.pe[:x.size(0), :].size())
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=2048, nhead=8, nlayers=6, dropout=0.2, embedding_weight=None):
        super(Transformer_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选
        self.attention = nn.MultiheadAttention(embed_dim=d_emb, num_heads=nhead)

        # 请自行设计分类器
        self.classifier = nn.Linear(d_emb, 15)  # 将隐藏层输出映射到类别的线性分类器

        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x) # 可选
        x = x.permute(1, 0, 2)  # 调整维度为 (seq_length, batch_size, d_emb)
        x, _ = self.attention(x, x, x)  # 自注意力机制
        x = x.permute(1, 0, 2)  # 恢复维度为 (batch_size, seq_length, d_emb)
        x = torch.mean(x, dim=1)  # 对序列维度进行平均池化
        x = self.classifier(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x


class Transformer_model_pre(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=2048, nhead=8, nlayers=6, dropout=0.2, embedding_weight=None):
        super(Transformer_model_pre, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 使用预训练的Transformer模型作为Embedding层
        self.embed = BertModel.from_pretrained('model/bert-base-chinese/')
        # 不训练,不需要计算梯度
        for param in self.embed.parameters():
            param.requires_grad_(False)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选
        self.attention = nn.MultiheadAttention(embed_dim=d_emb, num_heads=nhead)

        # 请自行设计分类器
        self.classifier = nn.Linear(d_emb, 15)  # 将隐藏层输出映射到类别的线性分类器

        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, input_ids, attention_mask, token_type_ids):
        # print(self.embed(x).size())
        x = self.embed(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)[0]
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x) # 可选
        x = x.permute(1, 0, 2)  # 调整维度为 (seq_length, batch_size, d_emb)
        x, _ = self.attention(x, x, x)  # 自注意力机制
        x = x.permute(1, 0, 2)  # 恢复维度为 (batch_size, seq_length, d_emb)
        x = torch.mean(x, dim=1)  # 对序列维度进行平均池化
        x = self.classifier(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x

    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=100, d_hid=80, nlayers=1, dropout=0.2, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)
        # print(self.embed.weight)

        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选
        self.relu = nn.ReLU()
        # 请自行设计分类器
        self.fc = nn.Linear(2 * d_hid * ntoken, 15)

        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)
        # print(x)
        x = self.lstm(x)[0]

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x).reshape(-1, 50*80*2)   # ntoken*nhid*2 (2 means bidirectional)
        x = self.relu(x)
        x = self.fc(x)

        #------------------------------------------------------end------------------------------------------------------#
        return x