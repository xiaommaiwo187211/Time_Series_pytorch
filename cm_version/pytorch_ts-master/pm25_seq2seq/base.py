import numpy as np
import pandas as pd


def create_features(df):
    year = pd.get_dummies(df['year'], drop_first=True, prefix='year')
    month = pd.get_dummies(df['month'], drop_first=True, prefix='month')
    df['date'] = df.year.astype(str) + '-' + df.month.astype(str).str.zfill(2) + '-' + df.day.astype(str).str.zfill(2)
    df['week_day'] = pd.to_datetime(df.date).dt.dayofweek
    week_day = pd.get_dummies(df['week_day'], drop_first=True, prefix='weekday')
    df['week_num'] = pd.to_datetime(df.date).dt.weekofyear
    df['sin_week'] = np.sin(2 * np.pi * df.week_num / 52)
    df['cos_week'] = np.cos(2 * np.pi * df.week_num / 52)
    df['sin_hour'] = np.sin(2 * np.pi * df.hour / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df.hour / 24)
    return pd.concat([df, year, month, week_day], axis=1)


def preprocessing(df, feature_cols, date_cols):
    df = df.copy()[feature_cols + date_cols]
    df = df.fillna(0)

    df['pm2.5'] = np.log1p(df['pm2.5'])
    df = pd.concat([df, df['pm2.5'].rename('target')], axis=1)
    df_train = df.iloc[:(-31 * 24), :].copy()
    df_test = df.iloc[-(31 * 24):, :].copy()

    Xtrain = df_train.iloc[:, :-1]
    ytrain = df_train.target.values.reshape((-1, 1))
    Xtest = df_test.iloc[:, :-1]
    ytest = df_test.target.values.reshape((-1, 1))

    mean_, std_ = Xtrain.iloc[:, 0].mean(), Xtrain.iloc[:, 0].std()
    for i in range(Xtrain.shape[1]):
        if Xtrain.columns[i] in date_cols:
            continue
        temp_mean = Xtrain.iloc[:, i].mean()
        temp_std = Xtrain.iloc[:, i].std()
        Xtrain.iloc[:, i] = (Xtrain.iloc[:, i] - temp_mean) / temp_std
        Xtest.iloc[:, i] = (Xtest.iloc[:, i] - temp_mean) / temp_std

    return Xtrain.values, ytrain, Xtest.values, ytest, mean_, std_
    # return Xtrain, ytrain, Xtest, ytest, 0, 1


def generate_samples(X, y, batch_size, input_seq_len, output_seq_len, seed=0):
    start_points = list(range(X.shape[0] - input_seq_len - output_seq_len + 1))
    # random_state = np.random.RandomState(seed)
    # start_points = random_state.permutation(start_points)
    start_points = np.random.permutation(start_points)
    iterations = int(np.ceil(len(start_points) / batch_size))

    for i in range(iterations):
        start_points_batch = start_points[i * batch_size:(i + 1) * batch_size]
        inputs_points_batch = [list(range(start_point, start_point + input_seq_len))
                               for start_point in start_points_batch]
        # sequence first batch second
        inputs_batch = np.take(X, inputs_points_batch, axis=0).transpose(1, 0, 2)

        outputs_points_batch = [list(range(start_point + input_seq_len, start_point + input_seq_len + output_seq_len))
                                for start_point in start_points_batch]
        outputs_batch = np.take(X, outputs_points_batch, axis=0).transpose(1, 0, 2)
        targets_batch = np.take(y, outputs_points_batch, axis=0).transpose(1, 0, 2)

        # inputs_batch: (input_sequence, batch, feature)
        # outputs_batch: (output_sequence, batch, feature)
        # targets_batch: (output_sequence, batch, 1)
        yield inputs_batch, outputs_batch, targets_batch, start_points_batch



if __name__ == '__main__':

    SEED = 1234

    np.random.seed(SEED)

    data = pd.read_csv('data/PRSA_data_2010.1.1-2014.12.31.xls')[24:]

    FEATURE_COLS = ['pm2.5']
    DUMMY_COL = []
    Xtrain, ytrain, Xtest, ytest, mean_, std_ = preprocessing(data, FEATURE_COLS, DUMMY_COL)

    INPUT_SEQ_LEN = 30
    OUTPUT_SEQ_LEN = 5
    BATCH_SIZE = 10

    inputs, outputs, targets, start_points = next(generate_samples(Xtrain, ytrain, BATCH_SIZE, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN))
    print(inputs)
