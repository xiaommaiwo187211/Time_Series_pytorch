import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TemporalConv(nn.Module):
	def __init__(self, input_channel_size, output_channel_size, kernel_size, padding, dilation):
		super(TemporalConv, self).__init__()
		self.padding = padding
		self.conv = nn.Conv1d(input_channel_size, output_channel_size,
							  kernel_size=kernel_size, padding=padding, dilation=dilation)

	def forward(self, inputs):
		outputs = self.conv(inputs)
		return outputs[:, :, :-self.padding]


class WaveNet(nn.Module):
	def __init__(self, encoder_seq_len, decoder_seq_len, input_channel_size, output_channel_size,
				 kernel_size, dilation_list, hidden_size, dropout, device):
		super(WaveNet, self).__init__()
		self.encoder_seq_len = encoder_seq_len
		self.decoder_seq_len = decoder_seq_len
		self.device = device
		tcn_list = []
		for dilation in dilation_list:
			padding = (kernel_size - 1) * dilation
			tcn_d = TemporalConv(input_channel_size, output_channel_size, kernel_size, padding, dilation)
			input_channel_size = output_channel_size
			tcn_list.append(tcn_d)
		self.tcn = nn.Sequential(*tcn_list)
		self.linear1 = nn.Linear(output_channel_size, hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(hidden_size, 1)

	def forward(self, inputs, teacher_forcing_ratio):

		seq_len = inputs.size(2)
		is_total_sequence = seq_len == self.encoder_seq_len + self.decoder_seq_len - 1
		if teacher_forcing_ratio > 0:
			assert is_total_sequence

		batch_size = inputs.size(0)
		decoder_inputs = inputs[:, :, :(-self.decoder_seq_len + 1)].clone() if is_total_sequence else inputs.clone()
		decoder_outputs = torch.zeros(batch_size, self.decoder_seq_len, device=self.device)
		for i in range(self.decoder_seq_len):
			# inputs: (batch, 1, encoder_sequence)
			# outputs: (batch, 1, output_channel)
			outputs = self.tcn(decoder_inputs).permute(0, 2, 1)[:, -1:, :]
			outputs = self.dropout(F.relu(self.linear1(outputs)))
			outputs = self.linear2(outputs)
			decoder_outputs[:, i] = outputs[:, 0, 0]

			if np.random.rand() < teacher_forcing_ratio:
				decoder_inputs = torch.cat([decoder_inputs, inputs[:, :, -self.decoder_seq_len + 1 + i].unsqueeze(2)], dim=2)
			else:
				decoder_inputs = torch.cat([decoder_inputs, outputs], dim=2)

		return decoder_outputs


		# if teacher_forcing_ratio == 1:
		# 	# inputs: (batch, 1, total_sequence)
		# 	# outputs: (batch, total_sequence, output_channel)
		# 	outputs = self.tcn(inputs).permute(0, 2, 1)
		# 	# outputs: (batch, total_sequence, hidden_size)
		# 	outputs = self.dropout(F.relu(self.linear1(outputs)))
		# 	# outputs: (batch, total_sequence, 1)
		# 	outputs = self.linear2(outputs)
		# 	return outputs[:, -self.decoder_seq_len:, :]

		# batch_size = inputs.size(0)
		# decoder_inputs = inputs.clone()
		# decoder_outputs = torch.zeros(batch_size, self.decoder_seq_len, device=self.device)
		# for i in range(self.decoder_seq_len):
		# 	outputs = self.tcn(decoder_inputs).permute(0, 2, 1)[:, -1:, :]
		# 	outputs = self.dropout(F.relu(self.linear1(outputs)))
		# 	outputs = self.linear2(outputs)
		# 	decoder_outputs[:, i] = outputs[:, 0, 0]
		# 	decoder_inputs = torch.cat([decoder_inputs, outputs], dim=2)
		# return decoder_outputs.unsqueeze(2)