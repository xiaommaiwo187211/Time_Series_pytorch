import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class EncoderCho(nn.Module):
    def __init__(self, embedding_size, inputs_size, encoder_hidden_size, layer_num, dropout, decoder_hidden_size):
        super(EncoderCho, self).__init__()
        self.gru = nn.GRU(embedding_size + inputs_size, encoder_hidden_size, layer_num, dropout=dropout)
        if layer_num == 1 and encoder_hidden_size == decoder_hidden_size:
            self.linear = nn.Sequential()
        else:
            self.linear = nn.Linear(encoder_hidden_size * layer_num, decoder_hidden_size)

    def forward(self, inputs, embedded):
        # inputs: (input_sequence, batch, feature)
		# embedded: (input_sequence, batch, embedded)
        # outputs: (output_sequence, batch, encoder_hidden)
        # hidden: (layer, batch, encoder_hidden)

        outputs, hidden = self.gru(torch.cat([inputs, embedded], dim=2))
        hidden = self.linear(hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)).unsqueeze(0)
        return hidden


class DecoderCho(nn.Module):
    def __init__(self, embedding_size, inputs_size, decoder_hidden_size, linear_hidden_size, linear_dropout):
        super(DecoderCho, self).__init__()
        self.gru = nn.GRU(embedding_size + inputs_size + decoder_hidden_size, decoder_hidden_size)
        self.linear = nn.Sequential(
            nn.Linear(embedding_size + inputs_size + decoder_hidden_size * 2, linear_hidden_size),
            nn.Dropout(linear_dropout),
            nn.Linear(linear_hidden_size, 1)
        )

    def forward(self, inputs, hidden, context, embedded):
        # inputs: (1, batch, feature)
        # context: (1, batch, decoder_hidden)
        # embedded: (1, batch, embedding)
        # hidden: (1, batch, decoder_hidden)
        # output: (batch, 1)

        output, hidden = self.gru(torch.cat([inputs, context, embedded], dim=2), hidden)
        output = F.softplus(self.linear(torch.cat([hidden, context, inputs, embedded], dim=2).squeeze(0)))
        return hidden, output


class Seq2SeqCho(nn.Module):
    def __init__(self, encoder, decoder, cate_size, embedding_size, device):
        super(Seq2SeqCho, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(cate_size, embedding_size)
        self.device = device

    def forward(self, cates, inputs, outputs, mean_std, teacher_forcing_ratio=0.5):
        # cates: (1, batch)
        # inputs: (input_sequence, batch, feature)
        # outputs: (output_sequence, batch, feature)
        # targets: (output_sequence, batch, 1)
        input_sequence_size = inputs.size(0)
        output_sequence_size, batch_size, feature_size = outputs.size()
        embedded = self.embedding(cates)
        # encoder_embedded: (input_sequence, batch, embedding)
        encoder_embedded = embedded.repeat(input_sequence_size, 1, 1)
        decoder_embedded = embedded.repeat(output_sequence_size, 1, 1)
        # encoder_hidden: (1, batch, decoder_hidden)
        encoder_hidden = self.encoder(inputs, encoder_embedded)

        decoder_input = torch.cat([inputs[-1:][:, :, :1], outputs[:1][:, :, 1:]], dim=2)
        encoder_context = encoder_hidden
        decoder_hidden = encoder_context.clone()
        decoder_outputs = torch.zeros(output_sequence_size, batch_size, 1).to(self.device)
        for i in range(output_sequence_size):
            decoder_hidden, decoder_output = self.decoder(decoder_input, decoder_hidden, encoder_context, decoder_embedded[i:i+1])
            decoder_outputs[i] = decoder_output

            if i == output_sequence_size - 1:
                break

            if np.random.random() < teacher_forcing_ratio:
                decoder_input = torch.cat([outputs[i:i+1][:, :, :1], outputs[i+1:i+2][:, :, 1:]], dim=2)
            else:
                # no teacher forcing
                decoder_output = (decoder_output - mean_std[:, :1]) / mean_std[:, 1:]
                decoder_input = torch.cat([decoder_output.unsqueeze(0), outputs[i+1:i+2][:, :, 1:]], dim=2)

        return decoder_outputs