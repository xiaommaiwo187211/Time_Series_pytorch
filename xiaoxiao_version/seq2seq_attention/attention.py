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