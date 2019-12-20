import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

from webtraffic_wavenet.data_loader import TimeSeriesDataSet
from webtraffic_wavenet.wavenet_model import WaveNet

import numpy as np
from time import time
from functools import partial



def train_epoch(model, data_loader, optimizer, loss_func, device):
    model.train()

    epoch_loss, total_num = 0, 0
    for i, (_, encoder_inputs, decoder_targets) in enumerate(data_loader):
        # encoder_inputs: (batch, 1, total_sequence)
        # decoder_targets: (batch, decoder_sequence, 1)

        optimizer.zero_grad()

        encoder_inputs = encoder_inputs.float().to(device).permute(0, 2, 1)
        decoder_targets = decoder_targets.float().to(device)

        model_outputs = model(encoder_inputs, teacher_forcing_ratio=1)
        loss, cnt = loss_func(model_outputs.contiguous().view(-1), decoder_targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * cnt
        total_num += cnt

    return epoch_loss / total_num


def evaluate_epoch(model, data_loader, loss_func, device):
    model.eval()

    epoch_loss, total_num = 0, 0
    with torch.no_grad():
        for i, (_, encoder_inputs, decoder_targets) in enumerate(data_loader):
            encoder_inputs = encoder_inputs.float().to(device).permute(0, 2, 1)
            decoder_targets = decoder_targets.float().to(device)

            model_outputs = model(encoder_inputs, teacher_forcing_ratio=0)
            loss, cnt = loss_func(model_outputs.contiguous().view(-1), decoder_targets.view(-1))
            epoch_loss += loss.item() * cnt
            total_num += cnt

    return epoch_loss / total_num


def mae_loss(outputs, targets, reduction='elementwise_mean'):
    mae = nn.L1Loss(reduction=reduction)
    return mae(outputs, targets), outputs.numel()


def save_embedding(target):
    embedding_weights = eval('seq2seq.embedding_' + target + '.weight.data.cpu().numpy()')
    embedding_index = eval('train_set.' + target + '_arr')
    embedding = np.concatenate([embedding_index.reshape((-1, 1)), embedding_weights], axis=1)
    np.save('model/' + target + '_embedding.npy', embedding)


def init_weights(m):
    for name, param in m.named_parameters():
        if len(param.data.size()) > 1:
            nn.init.kaiming_normal_(param.data)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def set_seed():
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    DATA_PATH = 'data/'
    ENCODER_SEQ_LEN = 430
    DECODER_SEQ_LEN = 60
    BATCH_SIZE = 128
    BATCH_SIZE_TEST = 1024
    EPOCH_NUM = 100
    LEARNING_RATE = 0.001
    INPUT_CHANNEL_SIZE = 1
    OUTPUT_CHANNEL_SIZE = 16
    INTERMEDIATE_CHANNEL_SIZE = 32
    POST_CHANNEL_SIZE = 128
    HIDDEN_SIZE = 128
    DROPOUT = 0.2
    KERNEL_SIZE = 2
    DILATION_LIST = [1, 2, 4, 8, 16, 32, 64, 128]

    set_seed()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    train_set = TimeSeriesDataSet(DATA_PATH, mode='train')
    test_set = TimeSeriesDataSet(DATA_PATH, mode='test')
    train_sampler = RandomSampler(train_set)
    test_sampler = RandomSampler(test_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_TEST, sampler=test_sampler, num_workers=0)

    seq2seq = WaveNet(ENCODER_SEQ_LEN, DECODER_SEQ_LEN, INPUT_CHANNEL_SIZE, OUTPUT_CHANNEL_SIZE,
                      INTERMEDIATE_CHANNEL_SIZE, KERNEL_SIZE, DILATION_LIST, POST_CHANNEL_SIZE, DROPOUT, device).to(device)

    seq2seq.apply(init_weights)

    optimizer = optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)

    min_val_loss = float('inf')
    not_descending_cnt = 0
    for epoch in range(EPOCH_NUM):

        start_time = time()

        train_loss = train_epoch(seq2seq, train_loader, optimizer, mae_loss, device)

        val_loss = evaluate_epoch(seq2seq, test_loader, mae_loss, device)

        end_time = time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('Epoch: %s | Time: %sm %ss' % (str(epoch + 1).zfill(2), epoch_mins, epoch_secs))
        print('\tTrain Loss: %.3f | Val Loss: %.3f' % (train_loss, val_loss))

        if val_loss >= min_val_loss:
            not_descending_cnt += 1
            if not_descending_cnt >= 20 and epoch >= 19 and epoch != EPOCH_NUM - 1:
                print('Early Stopped ...')
                break
        else:
            not_descending_cnt = 0
            if epoch >= 2:
                min_val_loss = val_loss
                torch.save(seq2seq.state_dict(), 'model/wavenet_model.pt')
                print()
                print('model saved with validation loss', val_loss)
                print()
