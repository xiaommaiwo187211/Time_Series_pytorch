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
	def __init__(self, input_channel_size, output_channel_size, kernel_size=2, dilation=1):
		self.padding = (kernel_size - 1) * dilation
		super(CausalDilatedConv, self).__init__(input_channel_size, output_channel_size, kernel_size,
												padding=self.padding, dilation=dilation)

	def forward(self, inputs):
		return super(CausalDilatedConv, self).forward(inputs)[:, :, :-self.padding[0]]
		### padding=3 means 往前补三个，往后补三个，那么我要causal我就要把最后三个干掉，不输出。


class TemporalConv(nn.Module):
	def __init__(self, input_channel_size, output_channel_size, intermediate_channel_size, kernel_size, dilation):
		super(TemporalConv, self).__init__()
		self.conv_dense_pre = DenseConv(input_channel_size, output_channel_size)  
		### (input_channel_size = 22, output_channel_size = 32), inpout_channel = feature_size
		### The first step：不需要causal
		self.conv_filter = CausalDilatedConv(output_channel_size, intermediate_channel_size,
							         		 kernel_size=kernel_size, dilation=dilation)
		### (32, 38, [2,1], dilation)
		self.conv_gate = CausalDilatedConv(output_channel_size, intermediate_channel_size,
							       		   kernel_size=kernel_size, dilation=dilation)
		### (32, 38, [2,1], dilation)
		self.conv_dense_post = DenseConv(intermediate_channel_size, output_channel_size)
		### (38, 32)

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()

	def forward(self, inputs):
		# inputs: (batch, 1, total_sequence)

		# outputs: (batch, output_channel, total_sequence)
		outputs = self.conv_dense_pre(inputs)
		### inputs = [batch = 128, channel = 22, time_step = 489]
		### outputs = [batch = 128, channel = 32, time_step = 489]
		# outputs_filter: (batch, intermediate_channel, total_sequence)
		outputs_filter = self.conv_filter(outputs)
		### outputs_filter = [batch = 128, channel =38, time_step =489]
		outputs_gate = self.conv_gate(outputs)
		### outputs_gate = [batch = 128, channel =38, time_step =489]
		outputs_post = self.tanh(outputs_filter) * self.relu(outputs_gate)
		### outputs_post = [batch = 128, channel =38, time_step =489]
		outputs_post = self.conv_dense_post(outputs_post)
		### outputs_post = [batch = 128, channel =32, time_step =489]
		outputs = outputs + outputs_post
		### outputs = [batch = 128, channel = 32, time_step = 489] + [batch = 128, channel =32, time_step =489]
		return outputs, outputs_post


class WaveNet(nn.Module):
	def __init__(self, encoder_seq_len, decoder_seq_len, input_channel_size, output_channel_size,
				 intermediate_channel_size, kernel_size, dilation_list, post_channel_size, dropout, device):
		super(WaveNet, self).__init__()
		self.encoder_seq_len = encoder_seq_len
		self.decoder_seq_len = decoder_seq_len
		self.device = device

		self.tcn_list = nn.ModuleList([TemporalConv(input_channel_size, output_channel_size, intermediate_channel_size,
								 					kernel_size, dilation=1)])
		### this part: firstly: inputs_channel_size = 22, 执行完第一步之后，output_channel_size = 32, 然后后面每一层都是32；

		### DILATION_LIST = [1, 2, 4, 8]
		for dilation in dilation_list[1:]:
			tcn_d = TemporalConv(output_channel_size, output_channel_size, intermediate_channel_size,
								 kernel_size, dilation)
			self.tcn_list.append(tcn_d)

		### 当构建完整个list后，直接可以通过 nn.sequential(*self.tcn_list)，然后在forward里写, self.tcn_list(inputs)即可。
		### 但是，由于现在我每一步还要进行一步操作，因此，必须要用nn.ModuleList([])将其包起来。

		self.conv_dense1 = DenseConv(output_channel_size, post_channel_size)
		### (output_channel_size, post_channel_size) =（32,142）
		self.dropout = nn.Dropout(dropout)
		### dropout =（0.2）
		self.conv_dense2 = DenseConv(post_channel_size, 1)
		### (32, 1)
		self.conv_post = nn.Sequential(self.conv_dense1,
									   self.dropout,
									   self.conv_dense2)

	def forward_t(self, inputs):
		outputs_post_list = []
		for tcn_d in self.tcn_list:
			outputs, outputs_post = tcn_d(inputs)
			### there is a problem: inputs = [128, 32, 489]; outputs = [128,32,489]; outputs_post = [128,32,489];
			### 一共5层，第一层到第二层：outputs = Conv(input) + 完整的Conv(inputs); outputs_post = 完整的Conv(inputs):resnet
			inputs = outputs
			### 上一层的outputs = 下一层的inputs
			outputs_post_list.append(outputs_post)
			### 将所有outputs_post append起来
		outputs = sum(outputs_post_list)
			### sum到一个outputs
		outputs = self.conv_post(outputs)
			### 将其转化为1个输出
			### outputs = [128,1,489]
		return outputs

	def forward(self, inputs, teacher_forcing_ratio):

		if teacher_forcing_ratio == 1:
			### train step: 不存在填值的问题，直接train 即可。
			# inputs: (batch, feature, total_sequence)
			# outputs: (batch, decoder_sequence, 1)
			outputs = self.forward_t(inputs).permute(0, 2, 1)
			return outputs[:, -self.decoder_seq_len:, :]

		### evaluate： INPUTS = [上一步的sale_qtty, others]
		batch_size = inputs.size(0)
		decoder_inputs = inputs[:, :, :-self.decoder_seq_len+1].clone()
		### 将inputs的encoder部分数据取出
		decoder_outputs = torch.zeros(batch_size, self.decoder_seq_len, device=self.device)
		### 存储所有outputs
		for i in range(self.decoder_seq_len):
			outputs = self.forward_t(decoder_inputs)[:, :, -1:]
			### 将encoder_inputs放进去：取最后一个outputs
			decoder_outputs[:, i] = outputs[:, 0, 0]
			### 将最后一个outputs的sale_qtty存起来
			decoder_exog = inputs[:, 1:, -self.decoder_seq_len + 1 + i].unsqueeze(2)
			### 取下一个time_step,然后特征：上一步的sale_qtty丢了
			decoder_exog = torch.cat([outputs, decoder_exog], dim=1)
			### 将下一步time_step的sale_qtty补齐
			decoder_inputs = torch.cat([decoder_inputs[:, :, 1:], decoder_exog], dim=2)
			### 将decoder_inputs的第一个time_step丢掉，补上下一步的time_step都补齐，进入下轮循环
		return decoder_outputs.unsqueeze(2)