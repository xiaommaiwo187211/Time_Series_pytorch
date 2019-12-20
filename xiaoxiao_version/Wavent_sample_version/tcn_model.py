import torch
import torch.nn as nn
import torch.nn.functional as F



class EncoderTCN(nn.Module):
	def __init__(self, dilation, input_channel_size=11, output_channel_size=11, kernel_size=2):
		super(EncoderTCN, self).__init__()
		self.conv1 = nn.Conv1d(input_channel_size, output_channel_size, kernel_size=kernel_size, dilation=dilation)
		self.bn1 = nn.BatchNorm1d(output_channel_size)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv1d(output_channel_size, output_channel_size, kernel_size=kernel_size, dilation=dilation)
		self.bn2 = nn.BatchNorm1d(output_channel_size)
		self.relu2 = nn.ReLU()

	def forward(self, inputs):
		outputs = self.relu1(self.bn1(self.conv1(inputs)))
		outputs = self.bn2(self.conv2(outputs))
		return self.relu2(outputs + inputs[:, :, -outputs.shape[2]:])


class DecoderMLP(nn.Module):
	def __init__(self, feature_size, conv_feature_size, hidden_size=64):
		super(DecoderMLP, self).__init__()
		self.fc1 = nn.Linear(feature_size, hidden_size)
		self.bn1 = nn.BatchNorm1d(hidden_size)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, conv_feature_size)
		self.bn2 = nn.BatchNorm1d(conv_feature_size)
		self.relu2 = nn.ReLU()

		self.fc3 = nn.Linear(22, 64)
		self.bn3 = nn.BatchNorm1d(64)
		self.relu3 = nn.ReLU()
		self.dropout = nn.Dropout(0.2)

	def forward(self, x_conv, x):
		out = self.relu1(self.bn1(self.fc1(x).permute(0, 2, 1)).permute(0, 2, 1))
		out = self.bn2(self.fc2(out).permute(0, 2, 1)).permute(0, 2, 1)
		out = self.relu2(x_conv + out)
		out = self.dropout(self.relu3(self.bn3(self.fc3(out).permute(0, 2, 1)).permute(0, 2, 1)))
		return out


class Seq2SeqDeepTCN(nn.Module):
	def __init__(self):
		super(Seq2SeqDeepTCN, self).__init__()
		self.store_embedding = nn.Embedding(370, 10)
		self.nMonth_embedding = nn.Embedding(12, 2)
		self.nYear_embedding = nn.Embedding(3, 2)
		self.mDay_embedding = nn.Embedding(31, 5)
		self.wday_embedding = nn.Embedding(7, 3)
		self.nHour_embedding = nn.Embedding(24, 4)
		self.holiday_embedding = nn.Embedding(2, 2)

		self.dilations = [1, 2, 4, 8, 16, 20, 32]
		encoder_tcn_list = []
		for dilation in self.dilations:
			encoder_tcn_list.append(EncoderTCN(dilation=dilation))
		self.encoder_tcn = nn.Sequential(*encoder_tcn_list)

		self.decoder_mlp = DecoderMLP(feature_size=28, conv_feature_size=22)

		self.Q10 = nn.Linear(64, 1)
		self.Q50 = nn.Linear(64, 1)
		self.Q90 = nn.Linear(64, 1)

	def forward(self, inputs, features):
		# preprocess
		store_embed = self.store_embedding(features[:, :, 0].long())
		embed_concat = torch.cat([
			store_embed,
			self.nYear_embedding(features[:, :, 2].long()),
			self.nMonth_embedding(features[:, :, 3].long()),
			self.mDay_embedding(features[:, :, 4].long()),
			self.wday_embedding(features[:, :, 5].long()),
			self.nHour_embedding(features[:, :, 6].long()),
			self.holiday_embedding(features[:, :, 7].long())],
			dim=2)
		input_store = store_embed[:, 0:1, :].repeat(1, 168, 1)
		output = torch.cat([input_store, inputs.view((inputs.shape[0], inputs.shape[1], 1))], dim=2)
		output = output.permute(0, 2, 1)
		for sub_tcn in self.encoder_tcn:
			output = sub_tcn(output)
		output = output.permute(0, 2, 1)
		output = output.contiguous().view(output.shape[0], 1, -1)
		output = output.repeat(1, 24, 1)
		output = self.decoder_mlp(output, embed_concat)
		output_Q10 = F.relu(self.Q10(output))
		output_Q10 = output_Q10.view(output_Q10.shape[0], -1)
		output_Q50 = F.relu(self.Q50(output))
		output_Q50 = output_Q50.view(output_Q50.shape[0], -1)
		output_Q90 = F.relu(self.Q90(output))
		output_Q90 = output_Q90.view(output_Q90.shape[0], -1)
		return output_Q10, output_Q50, output_Q90