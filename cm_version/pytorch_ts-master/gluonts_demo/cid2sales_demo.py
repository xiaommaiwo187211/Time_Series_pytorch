import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from itertools import islice
plt.style.use('ggplot')

import mxnet as mx

from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.util import to_pandas
from gluonts.distribution import NegativeBinomialOutput

from demo_utils import *

mx.random.seed(1)


TRAIN_START = '2018-01-01'
TRAIN_END = '2019-09-28'
TEST_START = '2019-06-28'
TEST_END = '2019-11-29'
TIME_FRAME = [TRAIN_START, TRAIN_END, TEST_START, TEST_END]


sales_df = pd.read_csv('../cid2sales_seq2seq/data/cid2_794_sales.csv')
sales_df.fillna({'sale_qtty': 0}, inplace=True)


cid3_cnt = len(set(sales_df.cid3))
cid3_dict = dict(zip(sorted(set(sales_df.cid3)), range(cid3_cnt)))

sales_df = generate_holiday(sales_df, TIME_FRAME)
global_redprice_std = sales_df.groupby('item_sku_id').redprice.std().mean()
global_nominal_netprice_std = sales_df.groupby('item_sku_id').nominal_netprice.std().mean()
global_instant_price_std = sales_df.groupby('item_sku_id').instant_price.std().mean()
global_price_std = np.mean([global_redprice_std, global_nominal_netprice_std, global_instant_price_std])


sku_list = sorted(set(sales_df.item_sku_id))
train_data, validation_data, test_data = [], [], []
for i in trange(len(sku_list)):
    sku = sku_list[i]
    train_listdataset, validation_listdataset, test_listdataset = generate_sku_listdataset(sku, sales_df, TIME_FRAME, cid3_dict, global_price_std)
    if len(train_listdataset['target']) < 93:
        continue
    train_data.append(train_listdataset)
    validation_data.append(validation_listdataset)
    test_data.append(test_listdataset)
train_data = ListDataset(train_data, freq='1D')
validation_data = ListDataset(validation_data, freq='1D')
test_data = ListDataset(test_data, freq='1D')


# train and validate
trainer = Trainer(epochs=20, batch_size=64, num_batches_per_epoch=2000, patience=1)
estimator = DeepAREstimator(freq="1D", prediction_length=62, context_length=93,
                            trainer=trainer,
                            use_feat_static_cat=True, cardinality=[cid3_cnt],
                            use_feat_dynamic_real=True,
                            distr_output=NegativeBinomialOutput())
predictor = estimator.train(training_data=train_data, validation_data=validation_data)


# inference
target_ind = 10
for i, (test_entry, forecast) in enumerate(islice(zip(validation_data, predictor.predict(test_data)), target_ind, target_ind+1)):
    sku = sku_list[target_ind]
    pred_df = plot_prediction(test_entry, forecast, sku)