import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DenseConv(nn.Conv1d):
	def __init__(self, input_channel_size, output_channel_size):
		super(DenseConv, self).__init__(input_channel_size, output_channel_size,
										kernel_size=1)
		self.relu = nn.ReLU()

	def forward(self, inputs):
		return self.relu(super(DenseConv, self).forward(inputs))


class CausalDilatedConv(nn.Conv1d):
	def __init__(self, input_channel_size, output_channel_size, kernel_size, dilation):
		self.padding = (kernel_size - 1) * dilation
		super(CausalDilatedConv, self).__init__(input_channel_size, output_channel_size, kernel_size,
												padding=self.padding, dilation=dilation)

	def forward(self, inputs):
		return super(CausalDilatedConv, self).forward(inputs)[:, :, :-self.padding[0]]


class TemporalConv(nn.Module):
	def __init__(self, input_channel_size, output_channel_size, intermediate_channel_size, kernel_size, dilation):
		super(TemporalConv, self).__init__()
		self.conv_dense_pre = DenseConv(input_channel_size, output_channel_size)
		self.conv_filter = CausalDilatedConv(output_channel_size, intermediate_channel_size,
							         		 kernel_size=kernel_size, dilation=dilation)
		self.conv_gate = CausalDilatedConv(output_channel_size, intermediate_channel_size,
							       		   kernel_size=kernel_size, dilation=dilation)
		self.conv_dense_post = DenseConv(intermediate_channel_size, output_channel_size)

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()

	def forward(self, inputs):
		# inputs: (batch, 1, total_sequence)

		# outputs: (batch, output_channel, total_sequence)
		outputs = self.conv_dense_pre(inputs)
		# outputs_filter: (batch, intermediate_channel, total_sequence)
		outputs_filter = self.conv_filter(outputs)
		outputs_gate = self.conv_gate(outputs)
		outputs_post = self.tanh(outputs_filter) * self.relu(outputs_gate)
		outputs_post = self.conv_dense_post(outputs_post)
		outputs = outputs + outputs_post
		return outputs, outputs_post


class WaveNet(nn.Module):
	def __init__(self, encoder_seq_len, decoder_seq_len, input_channel_size, output_channel_size,
				 intermediate_channel_size, kernel_size, dilation_list, post_channel_size, dropout, device):
		super(WaveNet, self).__init__()
		self.encoder_seq_len = encoder_seq_len
		self.decoder_seq_len = decoder_seq_len
		self.device = device

		self.tcn_list = nn.ModuleList()
		for dilation in dilation_list:
			tcn_d = TemporalConv(input_channel_size, output_channel_size, intermediate_channel_size,
								 kernel_size, dilation)
			self.tcn_list.append(tcn_d)

		self.conv_dense1 = DenseConv(output_channel_size, post_channel_size)
		self.dropout = nn.Dropout(dropout)
		self.conv_dense2 = DenseConv(post_channel_size, 1)
		self.conv_post = nn.Sequential(self.conv_dense1,
									   self.dropout,
									   self.conv_dense2)

	def forward_t(self, inputs):
		outputs_list = []
		for tcn_d in self.tcn_list:
			_, outputs = tcn_d(inputs)
			outputs_list.append(outputs)
		outputs = sum(outputs_list)
		outputs = self.conv_post(outputs)
		return outputs

	def forward(self, inputs, teacher_forcing_ratio):

		if teacher_forcing_ratio == 1:
			# inputs: (batch, 1, total_sequence)
			# outputs: (batch, decoder_sequence, 1)
			outputs = self.forward_t(inputs).permute(0, 2, 1)
			return outputs[:, -self.decoder_seq_len:, :]

		batch_size = inputs.size(0)
		decoder_inputs = inputs.clone()
		decoder_outputs = torch.zeros(batch_size, self.decoder_seq_len, device=self.device)
		for i in range(self.decoder_seq_len):
			outputs = self.forward_t(decoder_inputs)[:, :, -1:]
			decoder_outputs[:, i] = outputs[:, 0, 0]
			decoder_inputs = torch.cat([decoder_inputs[:, :, 1:], outputs], dim=2)
		return decoder_outputs.unsqueeze(2)

		# seq_len = inputs.size(2)
		# is_total_sequence = seq_len == self.encoder_seq_len + self.decoder_seq_len - 1
		# if teacher_forcing_ratio > 0:
		# 	assert is_total_sequence
		#
		# batch_size = inputs.size(0)
		# decoder_inputs = inputs[:, :, :(-self.decoder_seq_len + 1)].clone() if is_total_sequence else inputs.clone()
		# decoder_outputs = torch.zeros(batch_size, self.decoder_seq_len, device=self.device)
		# for i in range(self.decoder_seq_len):
		# 	# inputs: (batch, 1, encoder_sequence)
		# 	# outputs: (batch, 1, output_channel)
		# 	_, outputs = self.tcn(decoder_inputs)
		# 	outputs = self.linear(outputs)
		# 	decoder_outputs[:, i] = outputs[:, 0, 0]
		#
		# 	if np.random.rand() < teacher_forcing_ratio:
		# 		decoder_inputs = torch.cat([decoder_inputs, inputs[:, :, -self.decoder_seq_len + 1 + i].unsqueeze(2)], dim=2)
		# 	else:
		# 		decoder_inputs = torch.cat([decoder_inputs, outputs], dim=2)
		#
		# return decoder_outputs.unsqueeze(2)