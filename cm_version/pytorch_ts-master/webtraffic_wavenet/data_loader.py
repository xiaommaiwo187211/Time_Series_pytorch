import torch
from torch.utils.data import Dataset, Sampler, DataLoader, WeightedRandomSampler, RandomSampler

import numpy as np


class TimeSeriesDataSet(Dataset):
	def __init__(self, data_path, mode='train'):
		if mode == 'train':
			self.encoder_inputs = np.load(data_path + 'train_encoder_inputs.npy')[:5000]
			self.decoder_targets = np.load(data_path + 'train_decoder_targets.npy')[:5000]
		else:
			self.encoder_inputs = np.load(data_path + 'test_encoder_inputs.npy')[:5000]
			self.decoder_targets = np.load(data_path + 'test_decoder_targets.npy')[:5000]

	def __len__(self):
		return len(self.encoder_inputs)

	def __getitem__(self, item):
		return item, self.encoder_inputs[item, :], self.decoder_targets[item, :]