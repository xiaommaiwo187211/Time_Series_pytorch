import torch
from torch.utils.data import Dataset, Sampler, DataLoader, WeightedRandomSampler, RandomSampler

import pickle


class TimeSeriesDataSet(Dataset):
	def __init__(self, data_path, mode='train'):
		with open(data_path, 'rb') as f:
			trainX_dt, trainX2_dt, trainY_dt, trainY2_dt, testX_dt, testX2_dt, testY_dt, testY2_dt = pickle.load(f)
		if mode == 'train':
			self.sub_X, self.sub_Y = trainX_dt, trainY_dt
			self.future_Y = trainY2_dt
		else:
			self.sub_X, self.sub_Y = testX_dt, testY_dt
			self.future_Y = testY2_dt

	def __len__(self):
		return len(self.sub_X)

	def __getitem__(self, item):
		return item, self.sub_X[item, :], self.sub_Y[item, :], self.future_Y[item, :, :]