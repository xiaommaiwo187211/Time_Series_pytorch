import torch
from torch.utils.data import Dataset, Sampler, DataLoader, WeightedRandomSampler, RandomSampler

import numpy as np



class TimeSeriesDataSet(Dataset):
	def __init__(self, data_path, mode='train'):
		self.inputs = np.load(data_path + mode + '_inputs.npy')
		self.targets = np.load(data_path + mode + '_targets.npy')
		self.means = np.load(data_path + mode + '_means.npy')

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, item):
		# inputs, individuals, targets, means
		return item, self.inputs[item, :, :-1], self.inputs[item, 0, -1], self.targets[item], self.means[item]


# modified from torch.utils.data.WeightedRandomSampler
class WeightedSampler(Sampler):
	def __init__(self, data_path, replacement=True, mode='train'):
		means = np.load(data_path + mode + '_means.npy')
		self.weights = torch.tensor(means / means.sum(), dtype=torch.double)
		self.length = len(means)
		self.replacement = replacement

	def __len__(self):
		return self.length

	def __iter__(self):
		return iter(torch.multinomial(self.weights, self.length, self.replacement))



if __name__ == '__main__':
	DATA_PATH = './'
	BATCH_SIZE = 64

	train_set = TimeSeriesDataSet(DATA_PATH, mode='train')
	sampler = WeightedSampler(DATA_PATH, mode='train')  # Use weighted sampler instead of random sampler
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
	# inputs, individuals, targets, means = next(iter(train_loader))
	# print(inputs)

	test_set = TimeSeriesDataSet(DATA_PATH, mode='test')
	test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, sampler=RandomSampler(test_set), num_workers=4)
	inputs, individuals, targets, means = next(iter(test_loader))
	print(inputs)
