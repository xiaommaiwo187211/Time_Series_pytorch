import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_

from time import time
import numpy as np
import pandas as pd

from model import *
from base import *


def to_tensor(arr_list, device):
    tensor_list = []
    for arr in arr_list:
        tensor_list.append(torch.tensor(arr, device=device, dtype=torch.float))
    return tensor_list


def train_epoch(model, input_seq_len, output_seq_len, batch_size, optimizer, criterion, clip, device):
    # no need to use it here since there's no dropout or batchnorm
    model.train()

    epoch_loss, iterations = 0, 0
    for inputs, outputs, targets, start_points in generate_samples(Xtrain, ytrain, batch_size,
                                                                   input_seq_len, output_seq_len):
        optimizer.zero_grad()
        inputs, outputs, targets = to_tensor([inputs, outputs, targets], device)
        outputs = model(inputs, outputs, mean_, std_, teacher_forcing_ratio=0.3)
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

    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    ENCODER_HIDDEN_SIZE = 64
    DECODER_HIDDEN_SIZE = 128
    LAYER_NUM = 2
    ENCODER_DROPOUT = 0.5
    DECODER_DROPOUT = 0.5
    encoder = EncoderBahdanau(FEATURE_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)
    attn = AttentionBahdanau(ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)
    decoder = DecoderBahdanau(FEATURE_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, attn)
    seq2seq = Seq2SeqBahdanau(encoder, decoder, device).to(device)

    seq2seq.apply(init_weights)

    optimizer = optim.SGD(seq2seq.parameters(), lr=0.1, nesterov=True, momentum=0.9)
    criterion = nn.MSELoss(reduction='sum')

    INPUT_SEQ_LEN = 30
    OUTPUT_SEQ_LEN = 14
    BATCH_SIZE = 256
    EPOCH_NUM = 100
    CLIP = 1

    min_val_loss = float('inf')

    for epoch in range(EPOCH_NUM):

        start_time = time()

        train_loss = train_epoch(seq2seq, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN,
                                 BATCH_SIZE, optimizer, criterion, CLIP, device)

        set_seed()
        val_loss = evaluate(seq2seq, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN,
                            BATCH_SIZE, criterion, device)

        end_time = time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('Epoch: %s | Time: %sm %ss' % (str(epoch+1).zfill(2), epoch_mins, epoch_secs))
        print('\tTrain Loss: %.3f | Val Loss: %.3f' % (train_loss, val_loss))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(seq2seq.state_dict(), 'model/pm25_model.pt')
            print()
            print('model saved with validation loss', val_loss.item())
            print()

