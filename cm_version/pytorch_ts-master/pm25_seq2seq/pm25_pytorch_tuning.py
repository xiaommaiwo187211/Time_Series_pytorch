import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_

from time import time
import numpy as np
import pandas as pd
from functools import partial

from model import *
from base import *


def to_tensor(arr_list, device):
    tensor_list = []
    for arr in arr_list:
        tensor_list.append(torch.tensor(arr, device=device, dtype=torch.float))
    return tensor_list


def train_epoch(model, input_seq_len, output_seq_len, batch_size, optimizer, criterion, clip, tfr, device):
    # no need to use it here since there's no dropout or batchnorm
    model.train()

    epoch_loss, iterations = 0, 0
    for inputs, outputs, targets, start_points in generate_samples(Xtrain, ytrain, batch_size,
                                                                   input_seq_len, output_seq_len):
        optimizer.zero_grad()
        inputs, outputs, targets = to_tensor([inputs, outputs, targets], device)
        outputs = model(inputs, outputs, mean_, std_, teacher_forcing_ratio=tfr)
        if len(outputs) == 2:
            outputs = outputs[0]
        loss = criterion(outputs.view(-1), targets.view(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        iterations += 1

    # average batch loss
    # return epoch_loss / iterations

    # mse (set reduction to sum)
    total_num = output_seq_len * (len(Xtrain) - input_seq_len - output_seq_len + 1)
    return epoch_loss / total_num


def evaluate(model, input_seq_len, output_seq_len, batch_size, criterion, device):
    # no need to use it here since there's no dropout or batchnorm
    model.eval()

    epoch_loss, iterations = 0, 0
    with torch.no_grad():
        for inputs, outputs, targets, start_points in generate_samples(Xtest, ytest, batch_size,
                                                                       input_seq_len, output_seq_len):
            inputs, outputs, targets = to_tensor([inputs, outputs, targets], device)
            # turn off teacher forcing
            outputs = model(inputs, outputs, mean_, std_, teacher_forcing_ratio=0)
            if len(outputs) == 2:
                outputs = outputs[0]
            loss = criterion(outputs.view(-1), targets.view(-1))
            epoch_loss += loss
            iterations += 1

    # return epoch_loss / iterations
    total_num = output_seq_len * (len(Xtest) - input_seq_len - output_seq_len + 1)
    return epoch_loss / total_num


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m):
    for name, param in m.named_parameters():
        if len(param.data.size()) == 1:
            nn.init.normal_(param.data, 0, 1)
        else:
            nn.init.xavier_uniform_(param.data)


def set_seed():
    SEED = 12
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ind', type=int, default=1)
    parser.add_argument('--input_seq_len', type=int, default=30)
    parser.add_argument('--output_seq_len', type=int, default=14)
    parser.add_argument('--encoder_decoder_hidden_size', type=str, default='[64, 64]')
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--encoder_dropout', type=float, default=0.5)
    parser.add_argument('--decoder_dropout', type=float, default=0.5)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model_dir', type=str, default='/mnt/wangchengming5/jingpin/pytorch/model/pm25_model.pt')

    args = parser.parse_args()
    print(args)
    IND = args.ind
    INPUT_SEQ_LEN = args.input_seq_len
    OUTPUT_SEQ_LEN = args.output_seq_len
    ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE = eval(args.encoder_decoder_hidden_size)
    LAYER_NUM = args.layer_num
    ENCODER_DROPOUT = args.encoder_dropout
    DECODER_DROPOUT = args.decoder_dropout
    TFR = args.teacher_forcing_ratio
    BATCH_SIZE = args.batch_size
    EPOCH_NUM = args.epoch_num
    LR = args.lr
    MOMENTUM = args.momentum
    OPTIMIZER = partial(optim.SGD, lr=LR, nesterov=True, momentum=MOMENTUM) if args.optimizer == 'sgd' else partial(optim.Adam, lr=LR)
    MODEL_DIR = args.model_dir

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparam_dict_df = pd.read_csv('model/hyperparam_dict_df.csv')
    data = pd.read_csv('data/PRSA_data_2010.1.1-2014.12.31.xls')[24:]
    data = create_features(data)

    # pm2.5列必须放在第一个
    FEATURE_COLS = ['pm2.5', 'year', 'TEMP']
    DATE_COLS = ['sin_week', 'cos_week', 'sin_hour', 'cos_hour', 'month_2', 'month_3', 'month_4', 'month_5',
                 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
                 'month_12', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
                 'weekday_5', 'weekday_6']
    Xtrain, ytrain, Xtest, ytest, mean_, std_ = preprocessing(data, FEATURE_COLS, DATE_COLS)

    FEATURE_SIZE = len(FEATURE_COLS) + len(DATE_COLS)
    encoder = EncoderBahdanau(FEATURE_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)
    attn = AttentionBahdanau(ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)
    decoder = DecoderBahdanau(FEATURE_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, attn)
    seq2seq = Seq2SeqBahdanau(encoder, decoder, device).to(device)

    seq2seq.apply(init_weights)

    optimizer = OPTIMIZER(params=seq2seq.parameters())
    criterion = nn.MSELoss(reduction='sum')

    CLIP = 1

    min_val_loss = float('inf')
    not_descending_cnt = 0
    for epoch in range(EPOCH_NUM):

        start_time = time()

        train_loss = train_epoch(seq2seq, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN,
                                 BATCH_SIZE, optimizer, criterion, CLIP, TFR, device)

        val_loss = evaluate(seq2seq, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN,
                            BATCH_SIZE, criterion, device)

        end_time = time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('Epoch: %s | Time: %sm %ss' % (str(epoch + 1).zfill(2), epoch_mins, epoch_secs))
        print('\tTrain Loss: %.3f | Val Loss: %.3f' % (train_loss, val_loss))

        if val_loss >= min_val_loss:
            not_descending_cnt += 1
            if not_descending_cnt >= 15 and epoch >= 19 and epoch != EPOCH_NUM - 1:
                print('Early Stopped ...')
                break
        else:
            not_descending_cnt = 0
            if epoch >= 2:
                min_val_loss = val_loss
                torch.save(seq2seq.state_dict(), MODEL_DIR)
                print()
                print('model saved with validation loss', val_loss.item())
                print()
                hyperparam_dict_df.loc[IND, 'val_loss'] = val_loss.item()

    hyperparam_dict_df.to_csv('model/hyperparam_dict_df.csv', index=False)