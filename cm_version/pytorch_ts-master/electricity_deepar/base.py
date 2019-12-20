import pandas as pd
import numpy as np
from scipy.stats import zscore


# TODO: How to normalize features whose values are fixed in the future?
#       (example: date related features, monotonic features ...)
def create_date_features(df, start_index):
    df = df.copy()
    df['weekday'] = df.index.weekday.values
    df['hour'] = df.index.hour.values
    df['month'] = df.index.month.values
    df['distance'] = np.arange(len(df)) - start_index
    normalized = zscore(df[['weekday', 'hour', 'month']], axis=0)
    normalized_distance = np.concatenate([np.zeros(len(df[df.distance < 0])), zscore(df[df.distance >= 0].distance)], axis=0)
    normalized = np.concatenate([normalized_distance.reshape((-1, 1)), normalized], axis=1)
    return pd.DataFrame(normalized, index=df.index, columns=['distance', 'weekday', 'hour', 'month'])


def generate_features_df(df, time_frame):
    train_start, train_end, test_start, test_end = time_frame
    df = df.resample('1H', label='left', closed='right').sum()[train_start:test_end]
    start_index_list = (df.values != 0).argmax(axis=0).tolist()

    def generate_features_df_foreach(sample_index, start_index):
        features_df_date = create_date_features(df, start_index)
        # 7 columns: target_prev, distance, weekday, hour, month, individual, target
        features_df = pd.concat([df.iloc[:, sample_index].shift(1).rename('target_prev').fillna(0),
                                 features_df_date,
                                 pd.Series([sample_index for _ in range(len(df))], name='individual', index=df.index),
                                 df.iloc[:, sample_index].rename('target')], axis=1)
        features_df_train, features_df_test = features_df[train_start:train_end], features_df[test_start:test_end]
        Xtrain, ytrain = features_df_train.iloc[:, :-1], features_df_train.target.values
        Xtest, ytest = features_df_test.iloc[:, :-1], features_df_test.target.values
        return Xtrain.values, ytrain, Xtest.values, ytest

    # for every individual generate features
    Xtrain_list, ytrain_list, Xtest_list, ytest_list = [], [], [], []
    for i in range(df.shape[1]):
        Xtrain, ytrain, Xtest, ytest = generate_features_df_foreach(i, start_index_list[i])
        Xtrain_list.append(Xtrain)
        ytrain_list.append(ytrain)
        Xtest_list.append(Xtest)
        ytest_list.append(ytest)
    return Xtrain_list, ytrain_list, Xtest_list, ytest_list, start_index_list


# def generate_samples(X_list, y_list, start_index_list, input_seq_len, output_seq_len, mode='train'):
#
#     def generate_samples_foreach(sample_index):
#         X, y, start_index = X_list[sample_index], y_list[sample_index], start_index_list[sample_index]
#         if mode == 'train':
#             # TODO: include the previous 0 to allow the model to learn the behavior of new time series
#             start_points = len(X) - start_index - input_seq_len - output_seq_len + 1
#             inputs_points = [list(range(start_point, start_point + input_seq_len + output_seq_len)) for start_point in
#                              range(start_index, start_index + start_points)]
#         else:
#             start_points = len(X) - input_seq_len - output_seq_len + 1
#             inputs_points = [list(range(start_point, start_point + input_seq_len + output_seq_len)) for start_point in
#                              range(start_points)]
#         inputs_total = np.take(X, inputs_points, axis=0)
#         targets_total = np.take(y, inputs_points, axis=0)
#         target_mean_list = []
#         for i, feature_seq in enumerate(inputs_total):
#             non_zero_sum = (feature_seq[:input_seq_len, 0] > 0).sum(axis=0)
#             if non_zero_sum == 0:
#                 target_mean_list.append(0)
#                 continue
#             target_mean = feature_seq[:input_seq_len, 0].sum() / non_zero_sum + 1
#             inputs_total[i, :, 0] = feature_seq[:, 0] / target_mean
#             targets_total[i, :] = targets_total[i, :] / target_mean
#             target_mean_list.append(target_mean)
#         return inputs_total, targets_total, target_mean_list
#
#     inputs_list, targets_list, mean_list = [], [], []
#     for i in range(len(X_list)):
#         inputs, targets, means = generate_samples_foreach(i)
#         inputs_list.append(inputs)
#         targets_list.append(targets)
#         mean_list.append(means)
#     return np.concatenate(inputs_list), np.concatenate(targets_list), np.concatenate(mean_list)


def generate_samples(X_list, y_list, start_index_list, input_seq_len, output_seq_len, mode='train'):

    def generate_samples_foreach(sample_index):
        X, y, start_index = X_list[sample_index], y_list[sample_index], start_index_list[sample_index]
        # non-overlapping output sequence
        if mode == 'train':
            start_points = (len(X) - start_index - input_seq_len) // output_seq_len
            inputs_points = [list(range(start_index + start_point * output_seq_len, start_index + start_point * output_seq_len + input_seq_len + output_seq_len)) for start_point in
                             range(start_points)]
        else:
            start_points = (len(X) - input_seq_len) // output_seq_len
            inputs_points = [list(range(start_point * output_seq_len,
                                        start_point * output_seq_len + input_seq_len + output_seq_len))
                             for start_point in range(start_points)]
        # inputs_total: (sample, sequence, feature)
        # targets_total: (sample, sequence)
        inputs_total = np.take(X, inputs_points, axis=0)
        targets_total = np.take(y, inputs_points, axis=0)
        # for every sample of shape (sequence, feature)
        target_mean_list = []
        for i, feature_seq in enumerate(inputs_total):
            non_zero_sum = (feature_seq[:input_seq_len, 0] > 0).sum(axis=0)
            if non_zero_sum == 0:
                target_mean_list.append(0)
                continue
            target_mean = feature_seq[:input_seq_len, 0].sum() / non_zero_sum + 1
            # scaling
            inputs_total[i, :, 0] = feature_seq[:, 0] / target_mean
            # no need to scale test set targets
            targets_total[i, :] = targets_total[i, :] / target_mean if mode == 'train' else targets_total[i, :]
            target_mean_list.append(target_mean)
        return inputs_total, targets_total, np.array(target_mean_list)

    # for every individual generate samples
    inputs_list, targets_list, means_list = [], [], []
    for i in range(len(X_list)):
        inputs, targets, means = generate_samples_foreach(i)
        inputs_list.append(inputs)
        targets_list.append(targets)
        means_list.append(means)
    inputs_arr, targets_arr, means_arr = np.concatenate(inputs_list), np.concatenate(targets_list), np.concatenate(means_list)
    np.save(mode + '_inputs.npy', inputs_arr)
    np.save(mode + '_targets.npy', targets_arr)
    np.save(mode + '_means.npy', means_arr)
    return inputs_arr, targets_arr, means_arr




if __name__ == '__main__':

    # data = pd.read_csv('LD2011_2014.txt', sep=";", index_col=0, parse_dates=True, decimal=',')
    data = pd.read_csv('data_sub.csv', index_col=0, parse_dates=True)

    TRAIN_START, TRAIN_END = '2011-01-01 00:00:00', '2014-08-31 23:00:00'
    TEST_START, TEST_END = '2014-08-25 00:00:00', '2014-09-07 23:00:00'
    TIME_FRAME = (TRAIN_START, TRAIN_END, TEST_START, TEST_END)
    # encoder length is 7 days and decoder length is 1 day
    INPUT_SEQ_LEN, OUTPUT_SEQ_LEN = 168, 24

    Xtrain_li, ytrain_li, Xtest_li, ytest_li, start_index_li = generate_features_df(data, TIME_FRAME)
    inputs_arr, targets_arr, means_arr = generate_samples(Xtrain_li, ytrain_li, start_index_li, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, mode='train')
    _, _, _ = generate_samples(Xtest_li, ytest_li, start_index_li, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, mode='test')