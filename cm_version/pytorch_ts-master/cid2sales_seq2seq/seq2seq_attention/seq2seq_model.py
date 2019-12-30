import torch
import torch.nn as nn

import numpy as np


class EncoderCho(nn.Module):
    def __init__(self, embedding_size_cid3, encoder_inputs_size, encoder_hidden_size, decoder_hidden_size):
        super(EncoderCho, self).__init__()
        embedding_size = embedding_size_cid3
        self.gru = nn.GRU(embedding_size + encoder_inputs_size, encoder_hidden_size, bidirectional=True)
        self.linear = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, encoder_inputs, embedded):
        # encoder_inputs: (encoder_sequence, batch, feature)
        # embedded: (encoder_sequence, batch, embedded)
        # encoder_outputs: (encoder_sequence, batch, encoder_hidden * direction)
        # hidden: (layer * direction, batch, encoder_hidden)

        encoder_outputs, hidden = self.gru(torch.cat([encoder_inputs, embedded], dim=2))
        # hidden: (1, batch, encoder_hidden)
        hidden = self.tanh(self.linear(torch.cat([hidden[0], hidden[1]], dim=1))).unsqueeze(0)
        return encoder_outputs, hidden


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size, decoder_hidden_size)
        self.v = nn.Parameter(torch.rand(decoder_hidden_size))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, batch, decoder_hidden)
        # encoder_outputs: (encoder_sequence, batch, encoder_hidden * direction)
        # encoder_weights: (batch, encoder_sequence)

        encoder_seq_len, batch_size, _ = encoder_outputs.size()
        # decoder_hidden: (batch, encoder_sequence, decoder_hidden)
        decoder_hidden = decoder_hidden.repeat(encoder_seq_len, 1, 1).permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = self.tanh(self.attention(torch.cat([encoder_outputs, decoder_hidden], dim=2)))
        # energy: (batch, decoder_hidden, encoder_sequence)
        energy = energy.permute(0, 2, 1)
        # v: (batch, 1, decoder_hidden)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        encoder_weights = self.softmax(v.bmm(energy).squeeze(1))
        return encoder_weights


class DecoderCho(nn.Module):
    def __init__(self, embedding_size_cid3, decoder_inputs_size, decoder_hidden_size, encoder_hidden_size, attention):
        super(DecoderCho, self).__init__()
        embedding_size = embedding_size_cid3
        self.attention = attention
        self.gru = nn.GRU(embedding_size + decoder_inputs_size + encoder_hidden_size * 2, decoder_hidden_size)
        self.linear = nn.Linear(embedding_size + decoder_inputs_size + encoder_hidden_size * 2 + decoder_hidden_size, 1)

    def forward(self, decoder_inputs, decoder_hidden, encoder_outputs, embedded):
        # decoder_inputs: (1, batch, feature)
        # embedded: (1, batch, embedding)
        # decoder_hidden: (1, batch, decoder_hidden)
        # encoder_outputs: (encoder_sequence, batch, encoder_hidden * direction)
        # decoder_output: (batch, 1)

        # encoder_weights: (batch, 1, encoder_sequence)
        encoder_weights = self.attention(decoder_hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_weighted: (1, batch, encoder_hidden * direction)
        encoder_weighted = encoder_weights.bmm(encoder_outputs).permute(1, 0, 2)
        _, decoder_hidden = self.gru(torch.cat([decoder_inputs, embedded, encoder_weighted], dim=2), decoder_hidden)
        decoder_output = self.linear(torch.cat([decoder_inputs, embedded, encoder_weighted, decoder_hidden], dim=2))
        return decoder_hidden, decoder_output.squeeze(0), encoder_weights.squeeze(1)


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, cid3_size, embedding_size_cid3, dropout, device):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_cid3 = nn.Embedding(cid3_size, embedding_size_cid3)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, cid3, encoder_inputs, decoder_inputs, mean_std, teacher_forcing_ratio=0.5):
        # cid3: (1, batch)
        # encoder_inputs: (encoder_sequence, batch, feature)
        # decoder_inputs: (decoder_sequence, batch, feature)

        encoder_seq_len = encoder_inputs.size(0)
        decoder_seq_len, batch_size, feature_size = decoder_inputs.size()
        embedded = self.dropout(self.embedding_cid3(cid3))
        # encoder_embedded: (encoder_sequence, batch, embedding)
        encoder_embedded = embedded.repeat(encoder_seq_len, 1, 1)
        decoder_embedded = embedded.repeat(decoder_seq_len, 1, 1)
        # encoder_outputs: (decoder_sequence, batch, encoder_hidden * direction)
        # encoder_hidden: (layer, batch, encoder_hidden)
        encoder_outputs, encoder_hidden = self.encoder(encoder_inputs, encoder_embedded)

        decoder_input = torch.cat([encoder_inputs[-1:][:, :, :1], decoder_inputs[:1][:, :, 1:]], dim=2)
        decoder_hidden = encoder_hidden.clone()
        decoder_outputs = torch.zeros(decoder_seq_len, batch_size, 1).to(self.device)
        for i in range(decoder_seq_len):
            decoder_hidden, decoder_output, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                             decoder_embedded[i:i + 1])
            decoder_outputs[i] = decoder_output

            if i == decoder_seq_len - 1:
                break

            if np.random.random() < teacher_forcing_ratio:
                decoder_input = torch.cat([decoder_inputs[i:i + 1][:, :, :1], decoder_inputs[i + 1:i + 2][:, :, 1:]], dim=2)
            else:
                # no teacher forcing
                decoder_output = (decoder_output - mean_std[:, :1]) / mean_std[:, 1:]
                decoder_input = torch.cat([decoder_output.unsqueeze(0), decoder_inputs[i + 1:i + 2][:, :, 1:]], dim=2)

        return decoder_outputs