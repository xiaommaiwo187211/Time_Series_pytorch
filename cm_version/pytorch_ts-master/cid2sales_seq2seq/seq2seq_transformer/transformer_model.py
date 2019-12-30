import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CausalConv1D(nn.Conv1d):
    def __init__(self, input_channel_size, output_channel_size, kernel_size):
        self.padding = kernel_size - 1
        super(CausalConv1D, self).__init__(input_channel_size, output_channel_size, kernel_size, padding=self.padding)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = super(CausalConv1D, self).forward(inputs)[:, :, :-self.padding[0]] if self.padding[0] > 0 else super(CausalConv1D, self).forward(inputs)
        return outputs.permute(0, 2, 1)


class SelfAttention(nn.Module):
    def __init__(self, inputs_size, dk, dv, head_num, kernel_size, dropout):
        super(SelfAttention, self).__init__()

        assert inputs_size % head_num == 0

        self.inputs_size = inputs_size
        self.head_num = head_num
        self.dk = dk
        self.dv = dv

        self.conv_q = CausalConv1D(inputs_size, dk * head_num, kernel_size)
        self.conv_k = CausalConv1D(inputs_size, dk * head_num, kernel_size)
        self.linear_v = nn.Linear(inputs_size, dv * head_num)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, mask=None):
        # inputs: (batch, sequence, feature)
        # outputs: (batch, sequence, feature)

        batch_size = inputs.size(0)

        Q = self.conv_q(inputs)
        K = self.conv_k(inputs)
        V = self.linear_v(inputs)

        # Q: (batch, head_num, sequence, dk)
        Q = Q.view(batch_size, -1, self.head_num, self.dk).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.head_num, self.dk).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.head_num, self.dv).permute(0, 2, 1, 3)

        # energy: (batch, head_num, sequence, sequence)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.dk)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # attention: (batch, head_num, sequence, sequence)
        attention = self.dropout(torch.softmax(energy, dim=-1))
        outputs = torch.matmul(attention, V)
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.inputs_size)

        return outputs, attention


class PositionwiseFeedforward(nn.Module):
    def __init__(self, inputs_size, pf_hidden_size, dropout):
        super(PositionwiseFeedforward, self).__init__()

        self.linear1 = nn.Linear(inputs_size, pf_hidden_size)
        self.linear2 = nn.Linear(pf_hidden_size, inputs_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs: (batch, sequence, feature)
        # outputs: (batch, sequence, dv)

        outputs = self.dropout(self.relu(self.linear1(inputs)))
        outputs = self.linear2(outputs)
        return outputs


class TransFormerLayer(nn.Module):
    def __init__(self, inputs_size, dk, dv, head_num, kernel_size, pf_hidden_size, dropout):
        super(TransFormerLayer, self).__init__()
        self.self_attention = SelfAttention(inputs_size, dk, dv, head_num, kernel_size, dropout)
        self.positionwise_feedforward = PositionwiseFeedforward(inputs_size, pf_hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(inputs_size)

    def forward(self, inputs, mask):
        # inputs: (batch, sequence, feature)

        attention_outputs, _ = self.self_attention(inputs, mask)
        attention_outputs = self.layer_norm(inputs + self.dropout(attention_outputs))
        outputs = self.positionwise_feedforward(attention_outputs)
        outputs = self.layer_norm(attention_outputs + self.dropout(outputs))
        return outputs


class Transformer(nn.Module):
    def __init__(self, cid3_size, embedding_size_cid3,
                 inputs_size, layer_num, head_num, kernel_size, pf_hidden_size, dropout, device):
        super(Transformer, self).__init__()

        self.embedding_cid3 = nn.Embedding(cid3_size, embedding_size_cid3)
        embedding_size = embedding_size_cid3

        inputs_size = inputs_size + embedding_size

        dk = inputs_size // head_num
        dv = inputs_size // head_num
        self.transformer_list = nn.ModuleList([TransFormerLayer(inputs_size, dk, dv, head_num, kernel_size, pf_hidden_size, dropout)
                                               for _ in range(layer_num)])
        self.linear = nn.Linear(inputs_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward_t(self, inputs):
        seq_len = inputs.size(1)
        inputs_1 = inputs.clone()
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        for transformer in self.transformer_list:
            outputs = transformer(inputs_1, mask)
            inputs_1 = outputs.clone()
        outputs = self.linear(outputs)
        return outputs

    def forward(self, cid3, inputs, mean_std, decoder_seq_len=0, teacher_forcing=True):
        # inputs: (batch, sequence, feature)
        # outputs: (batch, sequence, 1)

        batch_size, seq_len, _ = inputs.size()
        embedded_cid3 = self.dropout(self.embedding_cid3(cid3)).unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat([inputs, embedded_cid3], dim=2)

        if teacher_forcing:
            outputs = self.forward_t(inputs)
            return outputs

        encoder_seq_len = seq_len - decoder_seq_len
        inputs_t = inputs[:, :encoder_seq_len + 1, :].clone()
        outputs = torch.zeros(batch_size, decoder_seq_len, device=self.device)
        for t in range(decoder_seq_len):
            outputs_t = self.forward_t(inputs_t)[:, -1:, :]
            outputs[:, t] = outputs_t[:, 0, 0]

            if t == decoder_seq_len - 1:
                break

            inputs_t_target = inputs_t[:, :, :1].clone()
            inputs_t_target = torch.cat([inputs_t_target, outputs_t], dim=1)
            inputs_t_exog = inputs[:, :encoder_seq_len + t + 2, 1:].clone()
            inputs_t = torch.cat([inputs_t_target, inputs_t_exog], dim=2)
        return outputs