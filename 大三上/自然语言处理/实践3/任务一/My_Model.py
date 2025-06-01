import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 max_seq_length, device, dictionary):
        """
        vocab_size: 词汇表的大小，表示模型将处理的不同词汇的数量。
        d_model: 模型的嵌入维度，表示模型将为每个词汇学习的向量的维度。
        nhead: 头的数量，表示多头自注意力机制中注意力头的个数。
        num_encoder_layers: 编码器层数，表示模型中编码器堆叠的层数。
        num_decoder_layers: 解码器层数，表示模型中解码器堆叠的层数。
        dim_feedforward: 每个位置的前馈网络的隐藏层维度。
        max_seq_length: 最大序列长度，表示输入序列的最大长度，用于生成位置编码。
        """
        super(TransformerModel, self).__init__()

        self.device = device  # 保存设备信息
        self.dictionary = dictionary

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers=num_encoder_layers
        )
        self.transformer_decoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers=num_decoder_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout()
        self.max_seq_length = max_seq_length

    def forward(self, src, tgt):
        src = self.embedding(src.to(self.device))
        src = self.positional_encoding(src.to(self.device))
        tgt = self.embedding(tgt.to(self.device))
        tgt = self.positional_encoding(tgt.to(self.device))
        # print(src.size())
        # print(tgt.size())

        src = self.transformer_encoder(src)
        tgt = self.transformer_decoder(tgt)

        tgt = self.dropout(tgt)  # 添加 dropout

        output = self.fc(tgt)

        return output

    def beam_search_decoding(self, output, beam_width=3, max_len=50):
        batch_size, seq_len, vocab_size = output.size()

        output = F.log_softmax(output.float(), dim=-1).to(self.device)  # 移动概率分布到指定设备

        decoded_outputs = []

        for b in range(batch_size):
            # 初始化每个样本的beam_width个假设
            beams = [{'score': 0, 'seq': [self.dictionary.word2tkn['[BOS]']]} for _ in range(beam_width)]

            for step in range(1, max_len + 1):
                candidates = []
                for beam in beams:
                    last_token = beam['seq'][-1]
                    if last_token == self.dictionary.word2tkn['[EOS]']:
                        candidates.append(beam)
                        continue

                    # 获取当前步骤的概率分布
                    prob_dist = output[b, step - 1, :]
                    topk_probs, topk_indices = torch.topk(prob_dist, beam_width, dim=-1)

                    for i in range(beam_width):
                        next_token = topk_indices[i].item()
                        new_beam = {
                            'score': beam['score'] + topk_probs[i].item(),
                            'seq': beam['seq'] + [next_token],
                        }
                        candidates.append(new_beam)

                # 选择得分最高的beam_width个假设
                candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_width]

            # 选择得分最高的假设作为输出
            best_beam = max(beams, key=lambda x: x['score'])
            decoded_output = torch.tensor(best_beam['seq']).unsqueeze(0)
            decoded_outputs.append(decoded_output)

        # 找到最长的序列长度
        max_seq_len = max(len(seq) for seq in decoded_outputs)

        # 填充或截断序列，使它们的长度一致
        padded_outputs = []
        for seq in decoded_outputs:
            seq = seq[0].tolist()
            if len(seq) >= 50:
                seq = seq[:50]
            else:
                for _ in range(max_len-len(seq)):
                    seq.append(0)
            padded_outputs.append(seq)


        # 将每个序列转换为张量
        padded_outputs = [torch.tensor(seq) for seq in padded_outputs]
        # 将所有样本的最优假设序列堆叠成一个张量
        decoded_outputs = torch.stack(padded_outputs, dim=0)

        return decoded_outputs
