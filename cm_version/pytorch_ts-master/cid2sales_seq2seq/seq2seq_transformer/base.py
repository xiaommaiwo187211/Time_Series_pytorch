import pandas as pd
import numpy as np
from scipy.stats import zscore
from tqdm import trange



# Feature flow for all skus
class Feature_Flow:

    def __init__(self, data_dir, time_frame, padding_size, normalize=True):
        self.df_total = pd.read_csv(data_dir + 'cid2_794_sales.csv')
        self.time_frame = time_frame
        self.padding_size = padding_size
        self.normalize = normalize

        self._preprocessing()

    def _preprocessing(self):
        train_start, _, _, test_end = self.time_frame
        self.df_total = self.df_total[self.df_total.date.between(train_start, test_end)]

    def create_date_features(self):
        train_start, _, _, test_end = self.time_frame
        train_start = str(pd.to_datetime(train_start) - pd.Timedelta(days=self.padding_size))[:10]
        date_df = pd.DataFrame(pd.date_range(train_start, test_end), columns=['date'])
        date_dt = date_df.date.dt
        date_df['month'] = date_dt.month
        date_df['weekday'] = date_dt.dayofweek
        date_df['date'] = date_df.date.astype(str)
        date_df = date_df[['date', 'month', 'weekday']]
        return date_df

    # observe holiday effects from cid2 aggregated sales trend
    def create_holiday_features(self, date_df):
        train_start, _, _, test_end = self.time_frame
        train_start = str(pd.to_datetime(train_start) - pd.Timedelta(days=self.padding_size))[:10]
        years = np.arange(int(train_start[:4]), int(test_end[:4]) + 1)
        # fixed holidays
        holidays_dict = {}

        for year in years:
            year = str(year) + '-'
            holidays_dict['bigpromo_pre'] = holidays_dict.get('bigpromo_pre', []) + \
                                            pd.date_range(year + '05-29', year + '05-31').astype(str).tolist() + \
                                            pd.date_range(year + '10-29', year + '10-31').astype(str).tolist() + \
                                            pd.date_range(year + '11-28', year + '11-30').astype(str).tolist()
            holidays_dict['bigpromo_begin'] = holidays_dict.get('bigpromo_begin', []) + \
                                              [year + '06-01'] + [year + '11-01'] + [year + '12-01']
            holidays_dict['bigpromo_middle'] = holidays_dict.get('bigpromo_middle', []) + \
                                               [year + '06-06'] + [year + '11-06'] + [year + '12-06']
            holidays_dict['bigpromo_climax'] = holidays_dict.get('bigpromo_climax', []) + \
                                               [year + '06-18'] + [year + '11-11'] + [year + '12-12']
            holidays_dict['promo618_period'] = holidays_dict.get('promo618_period', []) + \
                                               pd.date_range(year + '05-29', year + '06-18').astype(str).tolist()
            holidays_dict['promo1111_period'] = holidays_dict.get('promo1111_period', []) + \
                                                pd.date_range(year + '10-29', year + '11-11').astype(str).tolist()
            holidays_dict['promo1212_period'] = holidays_dict.get('promo1212_period', []) + \
                                                pd.date_range(year + '11-28', year + '12-12').astype(
                                                    str).tolist()
            holidays_dict['promo815'] = holidays_dict.get('promo815', []) + [year + '08-15']
            holidays_dict['promo315'] = holidays_dict.get('promo315', []) + [year + '03-15']
            holidays_dict['labor_day'] = holidays_dict.get('labor_day', []) + [year + '05-01']
            holidays_dict['national_day'] = holidays_dict.get('national_day', []) + [year + '10-01']

        # lunar holidays
        holidays_dict['spring_festival'] = pd.date_range('2018-02-15', '2018-02-22').astype(str).tolist() + \
                                           pd.date_range('2019-02-04', '2019-02-11').astype(str).tolist() + \
                                           pd.date_range('2020-01-24', '2020-01-31').astype(str).tolist() + \
                                           pd.date_range('2021-02-11', '2021-02-18').astype(str).tolist()

        for holiday in holidays_dict:
            holiday_df = pd.DataFrame(list(zip(holidays_dict[holiday], [1] * len(holidays_dict[holiday]))),
                                      columns=['date', holiday])
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

    def __init__(self, df, date_df, time_frame, padding_size, global_redprice_std, normalize=True, log_transform=True):
        self.df = df.sort_values('date').reset_index(drop=True)
        self.date_df = date_df
        self.time_frame = time_frame
        self.padding_size = padding_size
        self.normalize = normalize
        self.log_transform = log_transform
        self.global_redprice_std = global_redprice_std

        self.min_dt_df = self.df.date.min()

        self._preprocessing()

    def _preprocessing(self):
        # TODO: on_shelf days should fillna with 0 while off_shelf days should fillna with other values
        self.df.fillna({'sale_qtty': 0}, inplace=True)
        if self.log_transform:
            self.df['sale_qtty'] = np.log1p(self.df.sale_qtty)

    def padding_zero(self):
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
        train_start = self.time_frame[0]
        # train start date should be `lag` days after `train_start`
        max_dt_df = self.df.date.max()
        distance_arr = np.log1p(np.arange(len(pd.date_range(self.min_dt_df, max_dt_df))))
        if self.normalize:
            distance_arr = zscore(distance_arr)
        distance_arr = np.concatenate([np.zeros(len(self.df) - len(distance_arr)), distance_arr])
        self.df['distance_log'] = distance_arr
        self.df['instant_discount'] = self.df.instant_price / self.df.redprice
        self.df['nominal_discount'] = self.df.nominal_netprice / self.df.redprice
        self.df['sale_qtty_lag1'] = self.df.sale_qtty.shift(1).fillna(method='ffill').fillna(0)
        self.df['redprice_lag7'] = self.df.redprice.shift(7).fillna(method='ffill').fillna(method='bfill')
        self.df['redprice_diff_percent'] = (self.df.redprice - self.df.redprice.rolling(7).median().shift(
            1)) / self.df.redprice
        self.df['redprice_diff_percent'] = self.df.redprice_diff_percent.fillna(0)
        self.df.drop(['instant_price', 'nominal_netprice', 'netprice'], axis=1, inplace=True)
        # sale_qtty_lag1 must be the first column
        self.df = self.df[['sale_qtty_lag1'] + [col for col in self.df.columns if col != 'sale_qtty_lag1']]

    def create_train_test(self):
        if self.padding_size > 0:
            self.padding_zero()
        else:
            self.df.drop('is_padded', axis=1, inplace=True)
        self.generate_features()
        self.df['target'] = self.df['sale_qtty'].copy()

        train_start, train_end, test_start, test_end = self.time_frame
        train_data = self.df[self.df.date.between(train_start, train_end)]
        test_data = self.df[self.df.date.between(test_start, test_end)]
        cols_to_remove = ['item_sku_id', 'brand_code', 'cid3', 'date', 'target', 'sale_qtty']
        Xtrain, ytrain = train_data.drop(cols_to_remove, axis=1), train_data.target.values.reshape(
            (-1, 1))
        Xtest, ytest = test_data.drop(cols_to_remove, axis=1), test_data.target.values.reshape(
            (-1, 1))

        mean_, std_ = 0, 1
        if self.normalize:
            # mean_ = Xtrain.iloc[:, 0].mean()
            # std_ = Xtrain.iloc[:, 0].std()
            # std_ = std_ if std_ > 1e-2 else 1
            # Xtrain.iloc[:, 0] = (Xtrain.iloc[:, 0] - mean_) / std_
            # Xtest.iloc[:, 0] = (Xtest.iloc[:, 0] - mean_) / std_
            cols = ['instant_discount', 'redprice', 'redprice_diff_percent', 'redprice_lag7', 'nominal_discount']
            for col in cols:
                temp_mean = Xtrain[col].mean()
                temp_std = Xtrain[col].std()
                # for redprice and redprice_lag7 use global std instead of 1, otherwise there will be huge values
                if col in ['redprice', 'redprice_lag7']:
                    temp_std = self.global_redprice_std
                # replace std which is less than 0.01 with 1
                temp_std = temp_std if temp_std > 1e-2 else 1
                Xtrain[col] = (Xtrain[col] - temp_mean) / temp_std
                Xtest[col] = (Xtest[col] - temp_mean) / temp_std

        # nan is not allowed
        if (np.isnan(Xtrain).sum().sum() > 0) or (np.isnan(Xtest).sum().sum() > 0):
            raise ValueError("SKU %s contains nan values ..." % self.df.item_sku_id.min())

        return Xtrain.values, ytrain, Xtest.values, ytest, list(Xtrain.columns), mean_, std_

    def create_ts_samples(self, X, y, decoder_seq_len, skip_period, mode='train'):

        # number of start points considering skip period
        if mode == 'train':
            # series [1, 0, 3, 2, 4, 7, 2, 4, 3, 5, 6, 8, 4] with decoder_seq_len=5 and skip_period=4
            # will have 3 start points [1, 0, 3, 2, 4], [4, 7, 2, 4, 3], [3, 5, 6, 8, 4]
            start_points = list(
                range((len(X) - decoder_seq_len + skip_period) // skip_period))
        else:
            # test set only has one sequence for each sku
            start_points = [0]

        decoder_points = [list(range(start_point * skip_period,
                                     min(start_point * skip_period + decoder_seq_len, len(X))))
                         for start_point in start_points]

        # decoder_total: (sample, sequence, feature)
        # targets_total: (sample, sequence)
        # TODO: for test set, mask decoder sequence and target sequence
        decoder_total = np.take(X, decoder_points, axis=0)
        targets_total = np.take(y, decoder_points, axis=0)
        start_points = [start_point * skip_period for start_point in start_points]
        return decoder_total, targets_total, start_points


def main(data_dir, seed):
    feature_flow = Feature_Flow(data_dir, TIME_FRAME, PADDING_SIZE, NORMALIZE)
    features_df = feature_flow.create_features()
    global_redprice_std = features_df.groupby('item_sku_id').redprice.std().mean()
    date_df = feature_flow.create_date_features()
    date_df = feature_flow.create_holiday_features(date_df)
    rnd = np.random.RandomState(seed)
    item_sku_id_list = rnd.permutation(sorted(set(features_df.item_sku_id)))

    def main_sub(features_df_sub, mode):
        # SKU date range must be larger than (DECODER_SEQ_LEN - PADDING_SIZE)
        if features_df_sub.shape[0] <= DECODER_SEQ_LEN - PADDING_SIZE:
            return [], [], [], None, None
        feature_flow_sku = Feature_Flow_SKU(features_df_sub, date_df, TIME_FRAME, PADDING_SIZE, global_redprice_std,
                                            NORMALIZE, LOG_TRANSFORM)
        Xtrain, ytrain, Xtest, ytest, feature_cols, mean_, std_ = feature_flow_sku.create_train_test()
        X, y = eval('X' + mode), eval('y' + mode)
        decoder_total, targets_total, start_points = \
            feature_flow_sku.create_ts_samples(X, y, DECODER_SEQ_LEN, SKIP_PERIOD, mode)
        return decoder_total, targets_total, start_points, mean_, std_

    for mode in ['train', 'test']:
        decoder_arr, targets_arr, start_points_arr, mean_std_arr, sku_brand_cid3_arr = [], [], [], [], []
        for i in trange(len(item_sku_id_list)):
            item_sku_id = item_sku_id_list[i]
            features_df_sub = features_df[features_df.item_sku_id == item_sku_id]
            brand_code, cid3 = features_df_sub.brand_code.min(), features_df_sub.cid3.min()
            decoder_total, targets_total, start_points, mean_, std_ = main_sub(features_df_sub, mode)
            if len(start_points) == 0:
                continue
            # TODO: for test set, mask decoder sequence and target sequence
            if mode == 'test' and decoder_total.shape[1] < DECODER_SEQ_LEN:
                continue
            decoder_arr.append(decoder_total)
            targets_arr.append(targets_total)
            start_points_arr.append(start_points)
            mean_std_arr.append(np.array([mean_, std_] * len(start_points)).reshape((-1, 2)))
            sku_brand_cid3_arr.append(np.array([item_sku_id, brand_code, cid3] * len(start_points)).reshape((-1, 3)))

        decoder_arr = np.concatenate(decoder_arr)
        targets_arr, start_points_arr = np.concatenate(targets_arr), np.concatenate(start_points_arr)
        mean_std_arr, sku_brand_cid3_arr = np.concatenate(mean_std_arr), np.concatenate(sku_brand_cid3_arr)

        np.save(DATA_DIR + mode + '_decoder_inputs.npy', decoder_arr)
        np.save(DATA_DIR + mode + '_decoder_targets.npy', targets_arr)
        np.save(DATA_DIR + mode + '_start_points.npy', start_points_arr)
        np.save(DATA_DIR + mode + '_mean_std.npy', mean_std_arr)
        np.save(DATA_DIR + mode + '_sku_brand_cid3.npy', sku_brand_cid3_arr)

        # # in order for sku embedding, test data must not contain sku which isn't in train data
        # if mode == 'train':
        #     item_sku_id_list = list(set(sku_arr[:, 0]))




if __name__ == '__main__':
    DATA_DIR = '../data/'
    # decoder sequence length is 5 month
    TRAIN_START = '2018-01-01'
    TRAIN_END = '2019-09-28'
    TEST_START = '2019-06-28'
    TEST_END = '2019-11-29'
    TIME_FRAME = [TRAIN_START, TRAIN_END, TEST_START, TEST_END]
    DECODER_SEQ_LEN = 155
    PADDING_SIZE = 0
    SKIP_PERIOD = 21
    SEED = 1
    NORMALIZE = True
    LOG_TRANSFORM = True

    main(DATA_DIR, SEED)