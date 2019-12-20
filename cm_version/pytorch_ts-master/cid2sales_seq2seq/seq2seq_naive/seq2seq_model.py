import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class EncoderCho(nn.Module):
    def __init__(self, embedding_size_cid3, encoder_inputs_size, encoder_hidden_size, layer_num, dropout, decoder_hidden_size):
        super(EncoderCho, self).__init__()
        embedding_size = embedding_size_cid3
        self.gru = nn.GRU(embedding_size + encoder_inputs_size, encoder_hidden_size, layer_num, dropout=dropout)
        if layer_num == 1 and encoder_hidden_size == decoder_hidden_size:
            self.linear = nn.Sequential()
        else:
            self.linear = nn.Linear(encoder_hidden_size * layer_num, decoder_hidden_size)

    def forward(self, encoder_inputs, embedded):
        # encoder_inputs: (encoder_sequence, batch, feature)
        # embedded: (encoder_sequence, batch, embedded)
        # encoder_outputs: (decoder_sequence, batch, encoder_hidden)
        # hidden: (layer, batch, encoder_hidden)
        
        encoder_outputs, hidden = self.gru(torch.cat([encoder_inputs, embedded], dim=2))
        hidden = self.linear(hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)).unsqueeze(0)
        return hidden


class DecoderCho(nn.Module):
    def __init__(self, embedding_size_cid3, decoder_inputs_size, decoder_hidden_size, linear_hidden_size, linear_dropout):
        super(DecoderCho, self).__init__()
        embedding_size = embedding_size_cid3
        self.gru = nn.GRU(embedding_size + decoder_inputs_size + decoder_hidden_size, decoder_hidden_size)
        if linear_hidden_size == 0:
            self.linear = nn.Linear(embedding_size + decoder_inputs_size + decoder_hidden_size * 2, 1)
        else:
            self.linear = nn.Sequential(
                nn.Linear(embedding_size + decoder_inputs_size + decoder_hidden_size * 2, linear_hidden_size),
                nn.Dropout(linear_dropout),
                nn.Linear(linear_hidden_size, 1)
            )

    def forward(self, decoder_inputs, hidden, context, embedded):
        # decoder_inputs: (1, batch, feature)
        # context: (1, batch, decoder_hidden)
        # embedded: (1, batch, embedding)
        # hidden: (1, batch, decoder_hidden)
        # decoder_output: (batch, 1)

        decoder_output, hidden = self.gru(torch.cat([decoder_inputs, context, embedded], dim=2), hidden)
        decoder_output = F.softplus(self.linear(torch.cat([hidden, context, decoder_inputs, embedded], dim=2).squeeze(0)))
        return hidden, decoder_output


class Seq2SeqNaive(nn.Module):
    def __init__(self, encoder, decoder, cid3_size, embedding_size_cid3, dropout, device):
        super(Seq2SeqNaive, self).__init__()
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
        # encoder_embedded: (input_sequence, batch, embedding)
        encoder_embedded = embedded.repeat(encoder_seq_len, 1, 1)
        decoder_embedded = embedded.repeat(decoder_seq_len, 1, 1)
        # encoder_hidden: (1, batch, decoder_hidden)
        encoder_hidden = self.encoder(encoder_inputs, encoder_embedded)

        decoder_input = torch.cat([encoder_inputs[-1:][:, :, :1], decoder_inputs[:1][:, :, 1:]], dim=2)
        encoder_context = encoder_hidden
        decoder_hidden = encoder_context.clone()
        decoder_outputs = torch.zeros(decoder_seq_len, batch_size, 1).to(self.device)
        for i in range(decoder_seq_len):
            decoder_hidden, decoder_output = self.decoder(decoder_input, decoder_hidden, encoder_context,
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