import pandas as pd
import numpy as np
from scipy.stats import zscore
from tqdm import trange



# Feature flow for all skus
class Feature_Flow:

    def __init__(self, data_dir, time_frame, padding_size, normalize=True):
        self.df_total = pd.read_csv(data_dir)
        self.time_frame = time_frame
        self.padding_size = padding_size
        self.normalize = normalize

        self._preprocessing()

    def _preprocessing(self):
        train_start, _, _, test_end = self.time_frame
        self.df_total = self.df_total[self.df_total.date.between(train_start, test_end)]

    def create_date_features(self):
        def fourier_series(series, freq, mode='sin'):
            func = eval('np.' + mode)
            return func(2 * np.pi * series / freq)

        train_start, _, _, test_end = self.time_frame
        train_start = str(pd.to_datetime(train_start) - pd.Timedelta(days=self.padding_size))[:10]
        date_df = pd.DataFrame(pd.date_range(train_start, test_end), columns=['date'])
        date_dt = date_df.date.dt
        date_df['sin_month'] = fourier_series(date_dt.month, 12, 'sin')
        date_df['cos_month'] = fourier_series(date_dt.month, 12, 'cos')
        date_df['sin_weekday'] = fourier_series(date_dt.dayofweek, 7, 'sin')
        date_df['cos_weekday'] = fourier_series(date_dt.dayofweek, 7, 'cos')
        date_df['sin_weeknum'] = fourier_series(date_dt.weekofyear, 52, 'sin')
        date_df['cos_weeknum'] = fourier_series(date_dt.weekofyear, 52, 'cos')
        date_df['date'] = date_df.date.astype(str)
        date_df_cols = ['date', 'sin_month', 'cos_month', 'sin_weekday', 'cos_weekday', 'sin_weeknum', 'cos_weeknum']
        date_df = date_df[date_df_cols]

        if self.normalize:
            # normalize date features
            date_df = pd.concat([date_df.iloc[:, :1],
                                 pd.DataFrame(zscore(date_df.iloc[:, 1:], axis=0), columns=date_df_cols[1:])], axis=1)
        return date_df

    # observe holiday effects from cid3 aggregated sales trend
    # notes:
    # 315 家电节
    # 815 家电狂欢庆典
    # 919 电视影音超级品类日
    def create_holiday_features(self, date_df):
        train_start, _, _, test_end = self.time_frame
        train_start = str(pd.to_datetime(train_start) - pd.Timedelta(days=self.padding_size))[:10]
        years = np.arange(int(train_start[:4]), int(test_end[:4]) + 1)
        # fixed holidays
        holidays_dict = {}
        for year in years:
            year = str(year)
            holidays_dict['promo618'] = holidays_dict.get('promo618', []) + \
                                        pd.date_range(year + '-' + '06-01', year + '-' + '06-18').astype(str).tolist()
            holidays_dict['promo1111'] = holidays_dict.get('promo1111', []) + \
                                         pd.date_range(year + '-' + '11-01', year + '-' + '11-11').astype(str).tolist()
            holidays_dict['promo1212'] = holidays_dict.get('promo1212', []) + [year + '-' + '12-12']
            holidays_dict['promo0815'] = holidays_dict.get('promo0815', []) + [year + '-' + '08-15']
            holidays_dict['consume315'] = holidays_dict.get('consume315', []) + [year + '-' + '03-15']
            holidays_dict['labor_day'] = holidays_dict.get('labor_day', []) + [year + '-' + '05-01']
            holidays_dict['national_day'] = holidays_dict.get('national_day', []) + [year + '-' + '10-01']

        # lunar holidays
        holidays_dict['spring_festival'] = pd.date_range('2018-02-15', '2018-02-22').astype(str).tolist() + \
                                           pd.date_range('2019-02-04', '2019-02-11').astype(str).tolist() + \
                                           pd.date_range('2020-01-24', '2020-01-31').astype(str).tolist() + \
                                           pd.date_range('2021-02-11', '2021-02-18').astype(str).tolist()

        for holiday in holidays_dict:
            holiday_df = pd.DataFrame(list(zip(holidays_dict[holiday], [1] * len(holidays_dict[holiday]))), columns=['date', holiday])
            date_df = date_df.merge(holiday_df, on='date', how='left').fillna(0)

        return date_df

    def create_sales_median(self):
        train_start, train_end, _, _ = self.time_frame
        df_train = self.df_total[self.df_total.date.between(train_start, train_end)]
        df_train_sales = df_train.groupby('item_sku_id').sale_qtty.median(skipna=True).rename(
            'sale_qtty_median').reset_index()
        df_train_sales.fillna(0, inplace=True)
        if self.normalize:
            df_train_sales['sale_qtty_median'] = zscore(df_train_sales['sale_qtty_median'])
        return df_train_sales

    def create_features(self):
        date_df = self.create_date_features()
        date_df = self.create_holiday_features(date_df)
        df_train_sales = self.create_sales_median()
        df_total = self.df_total.merge(date_df, on='date') \
                                .merge(df_train_sales, on='item_sku_id')
        return df_total.sort_values(['item_sku_id', 'date']).reset_index(drop=True)


# Feature flow for one sku
class Feature_Flow_SKU:

    def __init__(self, df, date_df, time_frame, padding_size, lag_periods, normalize=True, log_transform=True):
        self.df = df.sort_values('date').reset_index(drop=True)
        self.date_df = date_df
        self.time_frame = time_frame
        self.padding_size = padding_size
        self.lag_periods = lag_periods
        self.normalize = normalize
        self.log_transform = log_transform

        self.min_dt_df = None

        self._preprocessing()

    def _preprocessing(self):
        # TODO: on_shelf days should fillna with 0 while off_shelf days should fillna with other values
        self.df.fillna({'sale_qtty': 0}, inplace=True)
        if self.log_transform:
            self.df['sale_qtty'] = np.log1p(self.df.sale_qtty)

    def padding_zero(self):
        self.min_dt_df = self.df.date.min()
        min_dt = str(pd.to_datetime(self.min_dt_df) - pd.Timedelta(days=self.padding_size))[:10]
        dt_list = pd.date_range(min_dt, self.min_dt_df)[:-1].astype(str)
        # default fill padding columns as zero
        padding_df = pd.DataFrame(np.zeros((self.padding_size, self.df.shape[1])), columns=self.df.columns)
        padding_df['item_sku_id'] = self.df.item_sku_id.min()
        padding_df['brand_code'] = self.df.brand_code.min()
        padding_df['cid3'] = self.df.cid3.min()
        padding_df['is_padded'] = 1
        padding_df['date'] = dt_list
        padding_df['sale_qtty_median'] = self.df.sale_qtty_median.min()
        padding_df['redprice'] = self.df.redprice[0]
        padding_df['instant_price'] = self.df.redprice[0]
        padding_df = padding_df.drop(self.date_df.columns[1:], axis=1).merge(self.date_df, on='date', how='left')
        padding_df = padding_df[self.df.columns]
        self.df = pd.concat([padding_df, self.df]).reset_index(drop=True)

    def generate_features(self):
        max_dt_df = self.df.date.max()
        distance_arr = np.arange(len(pd.date_range(self.min_dt_df, max_dt_df)))
        if self.normalize:
            distance_arr = zscore(distance_arr)
        distance_arr = np.concatenate([np.zeros(len(self.df) - len(distance_arr)), distance_arr])
        self.df['distance'] = distance_arr
        self.df['instant_discount'] = self.df.instant_price / self.df.redprice
        for t in self.lag_periods:
            self.df['sale_qtty_lag' + str(t)] = self.df.sale_qtty.shift(t) * 0.5
            # smooth around the lagging day
            if t > 1:
                self.df['sale_qtty_lag' + str(t)] += self.df.sale_qtty.shift(t-1) * 0.25 + self.df.sale_qtty.shift(t+1) * 0.25
            self.df['sale_qtty_lag' + str(t)] = self.df['sale_qtty_lag' + str(t)].fillna(0)
        self.df['redprice_lag7'] = self.df.redprice.shift(7).fillna(method='bfill')
        self.df['redprice_diff_percent'] = (self.df.redprice - self.df.redprice.rolling(7).median().shift(1)) / self.df.redprice
        self.df['redprice_diff_percent'] = self.df.redprice_diff_percent.fillna(0)
        self.df.drop(['instant_price', 'netprice'], axis=1, inplace=True)

    def create_train_test(self):
        self.padding_zero()
        self.generate_features()
        self.df['target'] = self.df['sale_qtty'].copy()

        train_start, train_end, test_start, test_end = self.time_frame
        train_data = self.df[self.df.date.between(train_start, train_end)]
        test_data = self.df[self.df.date.between(test_start, test_end)]
        cols_to_remove = ['item_sku_id', 'brand_code', 'cid3', 'date', 'target']
        Xtrain, ytrain = train_data.drop(cols_to_remove, axis=1), train_data.target.values.reshape(
            (-1, 1))
        Xtest, ytest = test_data.drop(cols_to_remove, axis=1), test_data.target.values.reshape(
            (-1, 1))

        mean_, std_ = 0, 1
        if self.normalize:
            mean_ = Xtrain.iloc[:, 0].mean()
            std_ = Xtrain.iloc[:, 0].std() if Xtrain.iloc[:, 0].sum() > 0 else 1
            cols = ['sale_qtty', 'instant_discount', 'redprice', 'redprice_diff_percent', 'redprice_lag7'] + \
                   ['sale_qtty_lag' + str(t) for t in self.lag_periods]
            for col in cols:
                temp_mean = Xtrain[col].mean()
                temp_std = Xtrain[col].std()
                if temp_std < 1e-4:
                    continue
                Xtrain[col] = (Xtrain[col] - temp_mean) / temp_std
                if col in Xtest:
                    Xtest[col] = (Xtest[col] - temp_mean) / temp_std

        # nan is not allowed
        if (np.isnan(Xtrain).sum().sum() > 0) or (np.isnan(Xtest).sum().sum() > 0):
            raise ValueError("SKU %s contains nan values ..." % self.df.item_sku_id.min())

        return Xtrain.values, ytrain, Xtest.values, ytest, mean_, std_, list(Xtrain.columns)

    def create_ts_samples(self, X, y, feature_cols, input_seq_len, output_seq_len, skip_period, mode='train'):
        def padding_nan(arr, mode):
            length = output_seq_len - arr.shape[1]
            if mode == 'train' or len(arr) == 0 or length == 0:
                return arr
            padded = np.zeros((arr.shape[0], length, arr.shape[2]))
            padded.fill(np.nan)
            arr_ = np.concatenate([arr, padded], axis=1)
            return arr_

        # number of start points considering skip period
        if mode == 'train':
            # series [1, 0, 3, 2, 4, 7, 2, 4, 3, 5, 6, 8, 4] with input_seq_len=3 and output_seq_len=2 and skip_period=4
            # will have 3 start points [1, 0, 3, 2, 4], [4, 7, 2, 4, 3], [3, 5, 6, 8, 4]
            start_points = list(
                range((len(X) - input_seq_len - output_seq_len + skip_period) // skip_period))
        else:
            # test set only has one sequence for each sku
            start_points = [0]
        inputs_points = [list(range(start_point * skip_period,
                                    start_point * skip_period + input_seq_len))
                         for start_point in start_points]
        outputs_points = [list(range(start_point * skip_period + input_seq_len,
                                     min(start_point * skip_period + input_seq_len + output_seq_len,
                                         len(X))))
                          for start_point in start_points]

        # columns to drop from decoder
        lag_cols = ['sale_qtty_lag' + str(t) for t in self.lag_periods]
        index_to_keep = [i for i in range(len(feature_cols)) if feature_cols[i] not in lag_cols]
        X_ = X[:, index_to_keep]

        # inputs_total: (sample, sequence, feature)
        # targets_total: (sample, sequence)
        inputs_total = np.take(X, inputs_points, axis=0)
        # for test set, pad output sequence and target sequence with nan if less than output_seq_len
        outputs_total = padding_nan(np.take(X_, outputs_points, axis=0), mode)
        targets_total = padding_nan(np.take(y, outputs_points, axis=0), mode)
        start_points = [start_point * skip_period for start_point in start_points]
        return inputs_total, outputs_total, targets_total, start_points


def main(data_dir, seed):
    feature_flow = Feature_Flow(data_dir, TIME_FRAME, PADDING_SIZE, NORMALIZE)
    features_df = feature_flow.create_features()
    date_df = feature_flow.create_date_features()
    date_df = feature_flow.create_holiday_features(date_df)
    rnd = np.random.RandomState(seed)
    item_sku_id_list = rnd.permutation(sorted(set(features_df.item_sku_id)))

    def main_sub(features_df_sub, mode):
        # SKU date range must be larger than (INPUT_SEQ_LEN - PADDING_SIZE)
        if features_df_sub.shape[0] <= INPUT_SEQ_LEN - PADDING_SIZE:
            return [], [], [], [], None, None
        feature_flow_sku = Feature_Flow_SKU(features_df_sub, date_df, TIME_FRAME, PADDING_SIZE, LAG_PERIODS, NORMALIZE, LOG_TRANSFORM)
        Xtrain, ytrain, Xtest, ytest, mean_, std_, feature_cols = feature_flow_sku.create_train_test()
        X, y = eval('X' + mode), eval('y' + mode)
        inputs_total, outputs_total, targets_total, start_points = feature_flow_sku.create_ts_samples(X, y, feature_cols,
                                                                                                      INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, SKIP_PERIOD, mode)
        return inputs_total, outputs_total, targets_total, start_points, mean_, std_


    for mode in ['train', 'test']:
        inputs_arr, outputs_arr, targets_arr, start_points_arr, mean_std_arr, sku_arr = [], [], [], [], [], []
        for i in trange(len(item_sku_id_list)):
            item_sku_id = item_sku_id_list[i]
            features_df_sub = features_df[features_df.item_sku_id == item_sku_id]
            brand_code, cid3 = features_df_sub.brand_code.min(), features_df_sub.cid3.min()
            inputs_total, outputs_total, targets_total, start_points, mean_, std_ = main_sub(features_df_sub, mode)
            if len(start_points) == 0:
                continue
            inputs_arr.append(inputs_total)
            outputs_arr.append(outputs_total)
            targets_arr.append(targets_total)
            start_points_arr.append(start_points)
            mean_std_arr.append(np.array([mean_, std_] * len(start_points)).reshape((-1, 2)))
            sku_arr.append(np.array([item_sku_id, brand_code, cid3] * len(start_points)).reshape((-1, 3)))

        inputs_arr, outputs_arr = np.concatenate(inputs_arr), np.concatenate(outputs_arr)
        targets_arr, start_points_arr = np.concatenate(targets_arr), np.concatenate(start_points_arr)
        mean_std_arr, sku_arr = np.concatenate(mean_std_arr), np.concatenate(sku_arr)

        np.save('data/' + mode + '_inputs.npy', inputs_arr)
        np.save('data/' + mode + '_outputs.npy', outputs_arr)
        np.save('data/' + mode + '_targets.npy', targets_arr)
        np.save('data/' + mode + '_start_points.npy', start_points_arr)
        np.save('data/' + mode + '_mean_std.npy', mean_std_arr)
        np.save('data/' + mode + '_sku.npy', sku_arr)




if __name__ == '__main__':
    DATA_DIR = 'data/cid2_794_sales.csv'
    # encoder length is 3 months and decoder length is 2 months
    TRAIN_START = '2018-01-01'
    TRAIN_END = '2019-09-01'
    TEST_START = '2019-06-01'
    TEST_END = '2019-11-02'
    TIME_FRAME = [TRAIN_START, TRAIN_END, TEST_START, TEST_END]
    INPUT_SEQ_LEN = 93
    OUTPUT_SEQ_LEN = 62
    PADDING_SIZE = 31
    SKIP_PERIOD = 15
    LAG_PERIODS = [7, 14, 31, 62]
    SEED = 1
    NORMALIZE = True
    LOG_TRANSFORM = True

    main(DATA_DIR, SEED)