import pandas as pd
from itertools import islice

import mxnet as mx

from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

from demo_utils import *


TRAIN_START, TRAIN_END = '2012-01-01 00:00:00', '2014-09-01 00:00:00'
TEST_START, TEST_END = '2014-08-25 00:00:00', '2014-09-08 00:00:00'

# df = pd.read_csv('data/LD2011_2014.txt', sep=";", index_col=0, parse_dates=True, decimal=',')
# df = df.resample('1H', label='left', closed='right').sum()[TRAIN_START:TEST_END]
# inds = (df.values != 0).mean(0) > 0.9
# df = df.iloc[:, inds]
# df.to_csv('electricity.csv', index=True)

df = pd.read_csv('data/electricity.csv')
cardinality = df.shape[1]


mx.random.seed(1)

# data preparation
train_data = ListDataset([{'start': TRAIN_START,
                           'target': df.loc[TRAIN_START:TRAIN_END].iloc[:, i].tolist(),
                           'feat_static_cat': [i]}
                          for i in range(cardinality)],
                         freq='1H')

test_data = ListDataset([{'start': TRAIN_START,
                          'target': df.loc[TRAIN_START:TEST_END].iloc[:, i].tolist(),
                          'feat_static_cat': [i]}
                         for i in range(cardinality)],
                        freq='1H')

# train
estimator = DeepAREstimator(freq="1H", prediction_length=24, context_length=168,
                            trainer=Trainer(epochs=20, batch_size=64),
                            use_feat_static_cat=True, cardinality=[cardinality])
# estimator = MQCNNEstimator(freq="1H", prediction_length=24, context_length=168,
#                            trainer=Trainer(epochs=20, batch_size=64), quantiles=[0.1, 0.5, 0.9])
# estimator = WaveNetEstimator(freq="1H", prediction_length=24,
#                              trainer=Trainer(epochs=20, batch_size=64), cardinality=[cardinality])

predictor = estimator.train(training_data=train_data, validation_data=test_data)

# inference
target_ind = 10
for i, (test_entry, forecast) in enumerate(islice(zip(test_data, predictor.predict(test_data)), target_ind, target_ind+1)):
    pred_df = plot_prediction(test_entry, forecast)
