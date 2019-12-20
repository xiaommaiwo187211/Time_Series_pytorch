import torch
import torch.nn as nn

import numpy as np


class EncoderCho(nn.Module):
    def __init__(self, embedding_size_cid3, encoder_inputs_size, encoder_hidden_size, layer_num):
        super(EncoderCho, self).__init__()
        embedding_size = embedding_size_cid3
        self.lstm = nn.LSTM(embedding_size + encoder_inputs_size, encoder_hidden_size, layer_num)

    def forward(self, encoder_inputs, embedded):
        # encoder_inputs: (encoder_sequence, batch, feature)
        # embedded: (input_sequence, batch, embedded)
        # encoder_outputs: (decoder_sequence, batch, encoder_hidden)
        # hidden: (layer, batch, encoder_hidden)
        
        encoder_outputs, (hidden, cell) = self.lstm(torch.cat([encoder_inputs, embedded], dim=2))
        return hidden, cell


class DecoderMLP(nn.Module):
    def __init__(self, embedding_size_cid3, decoder_inputs_size, encoder_hidden_size, decoder_hidden_size, decoder_seq_len, output_size):
        super(DecoderMLP, self).__init__()
        embedding_size = embedding_size_cid3
        self.global_mlp = nn.Linear(decoder_seq_len * (embedding_size + decoder_inputs_size) + encoder_hidden_size,
                                    (decoder_seq_len + 1) * decoder_hidden_size)
        self.local_mlp = nn.Linear(decoder_hidden_size * 2 + decoder_inputs_size + embedding_size, output_size)
        self.decoder_hidden_size = decoder_hidden_size

    def forward(self, decoder_inputs, encoder_context, embedded):
        # decoder_inputs: (output_sequence, batch, feature)
        # encoder_context: (1, batch, encoder_hidden)
        # embedded: (output_sequence, batch, embedding)
        # outputs: (output_sequence, batch, output_size)

        decoder_seq_len, batch_size, _ = decoder_inputs.size()
        decoder_inputs_cat = torch.cat([embedded, decoder_inputs], dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        encoder_context = encoder_context.squeeze(0)
        context = self.global_mlp(torch.cat([decoder_inputs_cat, encoder_context], dim=1))
        context_a = context[:, :self.decoder_hidden_size]
        context_c = context[:, self.decoder_hidden_size:]

        outputs = []
        for t in range(decoder_seq_len):
            inputs_t = torch.cat([decoder_inputs[t], embedded[t]], dim=1)
            context_ct = context_c[:, self.decoder_hidden_size*t:self.decoder_hidden_size*(t+1)]
            output_t = self.local_mlp(torch.cat([inputs_t, context_ct, context_a], dim=1))
            outputs.append(output_t)

        outputs = torch.stack(outputs, dim=0)
        return outputs


class Seq2SeqMQRNN(nn.Module):
    def __init__(self, encoder, decoder, cid3_size, embedding_size_cid3, dropout, device):
        super(Seq2SeqMQRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_cid3 = nn.Embedding(cid3_size, embedding_size_cid3)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, cid3, encoder_inputs, decoder_inputs):
        # cid3: (1, batch)
        # encoder_inputs: (encoder_sequence, batch, feature)
        # decoder_inputs: (decoder_sequence, batch, feature)

        input_sequence_size = encoder_inputs.size(0)
        output_sequence_size, batch_size, feature_size = decoder_inputs.size()
        embedded = self.dropout(self.embedding_cid3(cid3))
        # encoder_embedded: (input_sequence, batch, embedding)
        encoder_embedded = embedded.repeat(input_sequence_size, 1, 1)
        decoder_embedded = embedded.repeat(output_sequence_size, 1, 1)
        # encoder_hidden: (1, batch, decoder_hidden)
        encoder_hidden, encoder_cell = self.encoder(encoder_inputs, encoder_embedded)
        decoder_outputs = self.decoder(decoder_inputs, encoder_hidden, decoder_embedded)
        return decoder_outputs