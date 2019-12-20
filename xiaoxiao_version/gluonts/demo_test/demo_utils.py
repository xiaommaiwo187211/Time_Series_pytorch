import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.util import to_pandas


def plot_prediction(test_data_sample, test_data_pred_sample, plot_days=120, plot=True):
    test_data_sample = to_pandas(test_data_sample)[-plot_days:]
    dates = list(test_data_sample.index.astype(str))
    test_data_sample = test_data_sample.tolist()
    pred_len = test_data_pred_sample.samples.shape[1]
    fill_len = len(test_data_sample) - pred_len

    quantile_list = [0.05, 0.25, 0.5, 0.75, 0.95]
    pred_q_name, pred_q_list, pred_q_fill_list = [], [], []
    for q in quantile_list:
        q_str = str(int(q * 100)).zfill(2)
        pred_q = list(test_data_pred_sample.quantile(q))
        pred_q_fill = [np.nan] * fill_len + pred_q
        pred_q_name.append('pred_' + q_str)
        pred_q_list.append(pred_q)
        pred_q_fill_list.append(pred_q_fill)

    pred_df = pd.DataFrame([dates, test_data_sample] + pred_q_fill_list,
                           index=['date', 'real'] + pred_q_name).T

    if plot:
        pred_05, pred_25, pred_50, pred_75, pred_95 = pred_q_list
        x = list(range(len(dates)))
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(x, pred_df.real, label='target')
        plt.fill_between(x[-pred_len:], pred_05, pred_95, label='90% interval', color='cornflowerblue', alpha=0.3)
        plt.fill_between(x[-pred_len:], pred_25, pred_75, label='50% interval', color='cornflowerblue', alpha=0.5)
        plt.plot(x[-pred_len:], pred_50, label='pred', linewidth=2)
        plt.xticks(x[::7], dates[::7], rotation=90)
        plt.legend()

    return pred_df


def date_diff(date_start, date_end):
    return (pd.to_datetime(date_end) - pd.to_datetime(date_start)).days + 1


def generate_holiday(sales_df, time_frame):
    train_start, _, _, test_end = time_frame
    years = np.arange(int(train_start[:4]), int(test_end[:4]) + 1)
    # fixed holidays
    holidays_dict = {}
    for year in years:
        year = str(year) + '-'
        holidays_dict['bigpromo_pre'] = holidays_dict.get('bigpromo_begin', []) + \
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
    # monday to monday
    holidays_dict['spring_festival'] = pd.date_range('2018-02-13', '2018-02-18').astype(str).tolist() + \
                                       pd.date_range('2019-02-02', '2019-02-10').astype(str).tolist() + \
                                       pd.date_range('2020-01-21', '2020-01-26').astype(str).tolist() + \
                                       pd.date_range('2021-02-09', '2021-02-14').astype(str).tolist()

    for holiday in holidays_dict:
        holiday_df = pd.DataFrame(list(zip(holidays_dict[holiday], [1] * len(holidays_dict[holiday]))),
                                  columns=['date', holiday])
        sales_df = sales_df.merge(holiday_df, on='date', how='left').fillna(0)

    return sales_df


def generate_sku_listdataset(sku, sales_df, time_frame, cid3_dict):
    sales_df_sub = sales_df[sales_df.item_sku_id == sku].sort_values('date').reset_index(drop=True)
    start = sales_df_sub.date[0]
    cid3 = sales_df_sub.cid3[0]

    train_start, train_end, test_start, test_end = time_frame
    decoder_len = date_diff(train_end, test_end) - 1
    feat_cols = ['item_sku_id', 'date', 'sale_qtty', 'booking_flag', 'booking_pay_flag',
                 'presale_flag', 'presale_pay_flag', 'instant_flag', 'expose_flag', 'instant_hour',
                 'instant_price', 'redprice', 'nominal_netprice']
    sales_df_sub = sales_df_sub[feat_cols]
    train_data = sales_df_sub[sales_df_sub.date.between(train_start, train_end)]
    test_data = sales_df_sub[sales_df_sub.date.between(train_start, test_end)]
    cols_to_drop = ['item_sku_id', 'date', 'sale_qtty']
    Xtrain, ytrain = train_data.drop(cols_to_drop, axis=1), train_data.sale_qtty.values
    Xtest, ytest = test_data.drop(cols_to_drop, axis=1), test_data.sale_qtty.values

    cols_to_normalize = ['instant_price', 'redprice', 'nominal_netprice', 'instant_hour']
    for col in cols_to_normalize:
        temp_mean, temp_std = Xtrain[col].mean(), Xtrain[col].std()
        if temp_std < 1e-4:
            Xtrain[col] = Xtrain[col] - temp_mean
            Xtest[col] = Xtest[col] - temp_mean
            continue
        Xtrain[col] = (Xtrain[col] - temp_mean) / temp_std
        Xtest[col] = (Xtest[col] - temp_mean) / temp_std

    Xtrain, Xtest = Xtrain.values, Xtest.values

    train_listdataset = {'start': start,
                         'target': ytrain,
                         'feat_static_cat': cid3_dict[cid3],
                         'feat_dynamic_real': Xtrain.T}

    validation_listdataset = {'start': start,
                              'target': ytest,
                              'feat_static_cat': cid3_dict[cid3],
                              'feat_dynamic_real': Xtest.T}

    # test data's target should only contain dates before prediction_length days.
    # test data's feat_dynamic_real should contain dates for total_length days
    test_listdataset = {'start': start,
                        'target': ytest[:-decoder_len],
                        'feat_static_cat': cid3_dict[cid3],
                        'feat_dynamic_real': Xtest.T}

    return train_listdataset, validation_listdataset, test_listdataset
