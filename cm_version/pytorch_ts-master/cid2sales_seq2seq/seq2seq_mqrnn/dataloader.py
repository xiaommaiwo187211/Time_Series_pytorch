import torch
from torch.utils.data import Dataset, Sampler, DataLoader, WeightedRandomSampler, RandomSampler

import numpy as np
import pandas as pd


class TimeSeriesDataSet(Dataset):
    def __init__(self, data_path, mode='train'):
        self.encoder_inputs = np.load(data_path + mode + '_encoder_inputs.npy')
        self.decoder_inputs = np.load(data_path + mode + '_decoder_inputs.npy')
        self.decoder_targets = np.load(data_path + mode + '_decoder_targets.npy')
        self.start_points = np.load(data_path + mode + '_start_points.npy')
        self.mean_std = np.load(data_path + mode + '_mean_std.npy')

        sku_brand_cid3 = np.load(data_path + mode + '_sku_brand_cid3.npy')
        sku, self.sku_arr = self._convert_to_index(sku_brand_cid3[:, 0])
        brand, self.brand_arr = self._convert_to_index(sku_brand_cid3[:, 1])
        cid3, self.cid3_arr = self._convert_to_index(sku_brand_cid3[:, 2])
        self.sku_brand_cid3 = np.concatenate([sku, brand, cid3], axis=1)

    @staticmethod
    def _convert_to_index(arr):
        arr_set = set(arr)
        arr_dict = dict(zip(arr_set, np.arange(len(arr_set))))
        arr_index = pd.Series(arr).map(arr_dict).values.reshape((-1, 1))
        return arr_index, np.array(list(arr_set))

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, item):
        # item, encoder_inputs, decoder_inputs, decoder_targets, start_points, sku_brand_cid3
        return item, self.encoder_inputs[item], self.decoder_inputs[item], self.decoder_targets[item], self.start_points[item], self.sku_brand_cid3[item]