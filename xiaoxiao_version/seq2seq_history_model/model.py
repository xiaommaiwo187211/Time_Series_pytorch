import torch
import torch.nn as nn

import numpy as np


########################################################################################
# ------------------------------ Simple Seq2Seq Network ------------------------------ #
########################################################################################

class EncoderSimple(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(EncoderSimple, self).__init__()
        self.gru = nn.GRU(inputs_size, hidden_size)

    def forward(self, inputs):
        # inputs: (sequence, batch, feature)
        # outputs: (sequence, batch, hidden)
        # hidden: (layer, batch, hidden)

        outputs, hidden = self.gru(inputs)
        return hidden


class DecoderSimple(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(DecoderSimple, self).__init__()
        self.gru = nn.GRUCell(inputs_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, inputs, hidden):
        # inputs: (batch, feature)
        # hidden: (batch, hidden)
        # output: (batch, 1)

        hidden = self.gru(inputs, hidden)
        output = self.linear_1(hidden)
        output = self.linear_2(self.dropout(output))
        return hidden, output


class Seq2SeqSimple(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqSimple, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, inputs, outputs, mean_, std_, teacher_forcing_ratio=0.5):
        # inputs: (input_sequence, batch, feature)
        # outputs: (output_sequence, batch, feature)
        # targets: (output_sequence, batch, 1)
        output_sequence_size, batch_size, feature_size = outputs.size()
        encoder_hidden = self.encoder(inputs)

        decoder_input = torch.cat([inputs[-1][:, :1], outputs[0][:, 1:]], dim=1)
        decoder_hidden = encoder_hidden[0]
        decoder_outputs = torch.zeros(output_sequence_size, batch_size, 1).to(self.device)
        for i in range(output_sequence_size):
            decoder_hidden, decoder_output = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[i] = decoder_output

            if i == output_sequence_size - 1:
                break

            if np.random.random() < teacher_forcing_ratio:
                decoder_input = torch.cat([outputs[i][:, :1], outputs[i+1][:, 1:]], dim=1)
            else:
                # no teacher forcing
                decoder_output = (decoder_output - mean_) / std_
                decoder_input = torch.cat([decoder_output, outputs[i+1][:, 1:]], dim=1)

        return decoder_outputs




#######################################################################################################
# ------------------------------ Seq2Seq Network From 2014 Cho's Paper ------------------------------ #
#######################################################################################################

class EncoderCho(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(EncoderCho, self).__init__()
        self.gru = nn.GRU(inputs_size, hidden_size)

    def forward(self, inputs):
        # inputs: (sequence, batch, feature)
        # outputs: (sequence, batch, hidden)
        # hidden: (layer, batch, hidden)

        outputs, hidden = self.gru(inputs)
        return hidden


class DecoderCho(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(DecoderCho, self).__init__()
        self.gru = nn.GRUCell(inputs_size + hidden_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size * 2 + inputs_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs, hidden, context):
        # inputs: (batch, feature)
        # hidden: (batch, hidden)
        # output: (batch, 1)

        hidden = self.gru(torch.cat([inputs, context], dim=1), hidden)
        output = self.linear_1(torch.cat([hidden, context, inputs], dim=1))
        output = self.linear_2(self.dropout(output))
        return hidden, output


class Seq2SeqCho(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqCho, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, inputs, outputs, mean_, std_, teacher_forcing_ratio=0.5):
        # inputs: (input_sequence, batch, feature)
        # outputs: (output_sequence, batch, feature)
        # targets: (output_sequence, batch, 1)
        output_sequence_size, batch_size, feature_size = outputs.size()
        encoder_hidden = self.encoder(inputs)

        decoder_input = torch.cat([inputs[-1][:, :1], outputs[0][:, 1:]], dim=1)
        encoder_context = encoder_hidden[0]
        decoder_hidden = encoder_context.clone()
        decoder_outputs = torch.zeros(output_sequence_size, batch_size, 1).to(self.device)
        for i in range(output_sequence_size):
            decoder_hidden, decoder_output = self.decoder(decoder_input, decoder_hidden, encoder_context)
            decoder_outputs[i] = decoder_output

            if i == output_sequence_size - 1:
                break

            if np.random.random() < teacher_forcing_ratio:
                decoder_input = torch.cat([outputs[i][:, :1], outputs[i+1][:, 1:]], dim=1)
            else:
                # no teacher forcing
                decoder_output = (decoder_output - mean_) / std_
                decoder_input = torch.cat([decoder_output, outputs[i+1][:, 1:]], dim=1)

        return decoder_outputs




#############################################################################################################
# ------------------------------ Seq2Seq Network From 2014 Sutskever's Paper ------------------------------ #
#############################################################################################################

class EncoderSutskever(nn.Module):
    def __init__(self, inputs_size, hidden_size, layer_num, dropout=0.5):
        super(EncoderSutskever, self).__init__()
        self.gru = nn.GRU(inputs_size, hidden_size, layer_num, dropout=dropout)

    def forward(self, inputs):
        # inputs: (sequence, batch, feature)
        # outputs: (sequence, batch, hidden)
        # hidden: (layer, batch, hidden)

        outputs, hidden = self.gru(inputs)
        return hidden


class DecoderSutskever(nn.Module):
    def __init__(self, inputs_size, hidden_size, layer_num, dropout=0.5):
        super(DecoderSutskever, self).__init__()
        self.gru = nn.GRU(inputs_size, hidden_size, layer_num, dropout=dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size // 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_size // 4, 1)

    def forward(self, inputs, hidden):
        # inputs: (1, batch, feature)
        # outputs: (1, batch, hidden)
        # hidden: (layer, batch, hidden)

        output, hidden = self.gru(inputs, hidden)
        output = self.relu(self.linear_1(output.squeeze(0)))
        output = self.linear_2(self.dropout(output))
        return hidden, output


class Seq2SeqSutskever(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqSutskever, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, inputs, outputs, mean_, std_, teacher_forcing_ratio=0.5):
        # inputs: (input_sequence, batch, feature)
        # outputs: (output_sequence, batch, feature)
        # targets: (output_sequence, batch, 1)
        output_sequence_size, batch_size, feature_size = outputs.size()
        encoder_hidden = self.encoder(inputs)

        decoder_input = torch.cat([inputs[-1:][:, :, :1], outputs[:1][:, :, 1:]], dim=2)
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(output_sequence_size, batch_size, 1).to(self.device)
        for i in range(output_sequence_size):
            decoder_hidden, decoder_output = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[i] = decoder_output

            if i == output_sequence_size - 1:
                break

            if np.random.random() < teacher_forcing_ratio:
                decoder_input = torch.cat([outputs[i:i+1][:, :, :1], outputs[i+1:i+2][:, :, 1:]], dim=2)
            else:
                # no teacher forcing
                decoder_output = (decoder_output - mean_) / std_
                decoder_input = torch.cat([decoder_output.unsqueeze(0), outputs[i+1:i+2][:, :, 1:]], dim=2)
        return decoder_outputs




###########################################################################################################
# ------------------------------ Seq2Seq Network with Bahdanau's Attention ------------------------------ #
###########################################################################################################

class EncoderBahdanau(nn.Module):
    def __init__(self, inputs_size, encoder_hidden_size, decoder_hidden_size):
        super(EncoderBahdanau, self).__init__()
        self.gru = nn.GRU(inputs_size, encoder_hidden_size, bidirectional=True)
        self.linear = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

    def forward(self, inputs):
        # inputs: (sequence, batch, feature)
        # outputs: (sequence, batch, hidden * direction)
        # hidden: (layer * direction, batch, hidden)

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        outputs, hidden = self.gru(inputs)
        hidden = torch.tanh(self.linear(torch.cat((hidden[::2, :, :], hidden[1::2, :, :]), dim=2)))
        return outputs, hidden


class AttentionBahdanau(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(AttentionBahdanau, self).__init__()
        self.attn = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size, decoder_hidden_size)
        self.v = nn.Parameter(torch.rand(decoder_hidden_size), requires_grad=True)

    def forward(self, decoder_hidden, encoder_outputs):
        # encoder outputs: (encoder_sequence, batch, encoder_hidden * direction)
        # decoder hidden: (layer, batch, decoder_hidden)
        # attention: (batch, encoder_sequence)
        encoder_seq_len, batch_size, _ = encoder_outputs.size()
        decoder_hidden = decoder_hidden.repeat(encoder_seq_len, 1, 1)
        # pass encoder outputs and previous decoder hidden through a multi-layer perceptron, this is called an alignment model
        # e(i, j) = a(s(i-1), h(j))
        # energy scores how well  the inputs around position j and the outputs around position i match
        energy = torch.tanh(self.attn(torch.cat([encoder_outputs, decoder_hidden], dim=2)))
        # bmm requires batch first
        energy = energy.permute(1, 2, 0)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.softmax(torch.bmm(v, energy).squeeze(1), dim=1)
        return attention


class DecoderBahdanau(nn.Module):
    def __init__(self, inputs_size, encoder_hidden_size, decoder_hidden_size, attention):
        super(DecoderBahdanau, self).__init__()
        self.gru = nn.GRU(inputs_size + encoder_hidden_size * 2, decoder_hidden_size)
        self.attention = attention
        self.linear = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size + inputs_size, 1)

    def forward(self, inputs, decoder_hidden, encoder_outputs):
        # inputs: (1, batch, feature)
        # decoder hidden: (1, batch, decoder_hidden)
        # encoder outputs: (sequence, batch, encoder_hidden * direction)
        attn_weights = self.attention(decoder_hidden, encoder_outputs)
        # bmm requires batch first
        attn_weights = attn_weights.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_weighted = torch.bmm(attn_weights, encoder_outputs).permute(1, 0, 2)
        output, hidden = self.gru(torch.cat([inputs, attn_weighted], dim=2), decoder_hidden)
        output = self.linear(torch.cat([inputs.squeeze(0), attn_weighted.squeeze(0), output.squeeze(0)], dim=1))
        return hidden, output, attn_weights.squeeze(1)


class Seq2SeqBahdanau(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqBahdanau, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, inputs, outputs, mean_, std_, teacher_forcing_ratio=0.5):
        # inputs: (input_sequence, batch, feature)
        # outputs: (output_sequence, batch, feature)
        # targets: (output_sequence, batch, 1)
        input_sequence_size = inputs.size(0)
        output_sequence_size, batch_size, feature_size = outputs.size()
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        decoder_input = torch.cat([inputs[-1:][:, :, :1], outputs[:1][:, :, 1:]], dim=2)
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(output_sequence_size, batch_size, 1).to(self.device)
        attentions = torch.zeros(output_sequence_size, batch_size, input_sequence_size).to(self.device)
        for i in range(output_sequence_size):
            decoder_hidden, decoder_output, attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[i] = decoder_output
            attentions[i] = attention

            if i == output_sequence_size - 1:
                break

            if np.random.random() < teacher_forcing_ratio:
                decoder_input = torch.cat([outputs[i:i+1][:, :, :1], outputs[i+1:i+2][:, :, 1:]], dim=2)
            else:
                # no teacher forcing
                decoder_output = (decoder_output - mean_) / std_
                decoder_input = torch.cat([decoder_output.unsqueeze(0), outputs[i+1:i+2][:, :, 1:]], dim=2)
        return decoder_outputs, attentions