import sys
import numpy as np
import pandas as pd
from time import time
from scipy.stats import randint as sp_randint, uniform as sp_uniform
import das.udf.estimator as de


ORIGINAL_STDOUT = sys.stdout


def generate_hyperparam_list(hyperparam_list_dict, n_iter=5, seed=0):
    rnd = np.random.RandomState(seed)
    items = sorted(hyperparam_list_dict.items())
    hyperparam_list = []
    for _ in range(n_iter):
        hyperparam_dict = dict()
        for k, v in items:
            if hasattr(v, "rvs"):
                hyperparam_dict[k] = v.rvs(random_state=rnd)
            else:
                hyperparam_dict[k] = v[rnd.randint(len(v))]
        hyperparam_list.append(hyperparam_dict)
    return hyperparam_list


def get_val_loss(ind):
    hyperparam_dict_df = pd.read_csv('model/hyperparam_dict_df.csv')
    return hyperparam_dict_df.loc[ind, 'val_loss']


def run_job(hyperparam_list_dict, n_iter=5, seed=12):
    sys.stdout = ORIGINAL_STDOUT
    hyperparam_list = generate_hyperparam_list(hyperparam_list_dict, n_iter, seed)

    hyperparam_dict_df = pd.DataFrame([[i + 1, hyperparam_dict, 0]
                                       for i, hyperparam_dict in enumerate(hyperparam_list)],
                                      columns=['ind', 'param_dict', 'val_loss'])
    hyperparam_dict_df.to_csv('model/hyperparam_dict_df.csv', index=False)

    for i, hyperparam_dict in enumerate(hyperparam_list):
        hyperparam_dict.update({'ind': i})
        i += 1
        start = time()
        print('%s' % hyperparam_dict)
        sys.stdout = open("model/model_log%s.txt" % str(i).zfill(2), "w+")
        u_job = de.Udf(entry_point='/mnt/wangchengming5/jingpin/pytorch/pm25_pytorch_tuning.py',
                       image_name='repo.jd.local/public/notebook:nb5.5-pytorch0.4-py3-gpu',
                       hyperparameters=hyperparam_dict,
                       model_dir='/mnt/wangchengming5/jingpin/pytorch/model/pm25_model%s.pt' % str(i).zfill(2),
                       train_gpu_count=1)
        u_job.fit(base_job_name='pytorch-ts')
        sys.stdout.close()
        sys.stdout = ORIGINAL_STDOUT
        print('Val Loss: %.3f    Time Spent: %ds\n' % (get_val_loss(i - 1), time() - start))
    return hyperparam_list




if __name__ == '__main__':
    hyperparam_list_dict = {
        'batch_size': [128],
        'encoder_decoder_hidden_size': ['[64, 64]', '[64, 128]', '[128, 128]'],
        'lr': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        'momentum': [0.9, 0.5],
        'teacher_forcing_ratio': [0.3, 1]
    }

    hyperparam_list = run_job(hyperparam_list_dict, n_iter=5)