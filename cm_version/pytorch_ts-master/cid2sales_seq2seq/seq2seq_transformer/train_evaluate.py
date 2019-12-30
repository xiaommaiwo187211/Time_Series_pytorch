import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

from seq2seq_transformer.dataloader import WeightedSampler, TimeSeriesDataSet
from seq2seq_transformer.transformer_model import *

import numpy as np
from time import time




def train_epoch(model, data_loader, optimizer, loss_func, clip, device):
    model.train()

    epoch_loss, total_num = 0, 0
    for i, (_, decoder_inputs, decoder_targets, sku_start_points, mean_std, sku_brand_cid3) in enumerate(data_loader):
        # decoder_inputs: (batch, sequence, feature)
        # decoder_targets: (batch, sequence, 1)
        # sku_brand_cid3: (batch, 3)

        optimizer.zero_grad()

        decoder_inputs = decoder_inputs.float().to(device)
        decoder_targets = decoder_targets.float().to(device)
        mean_std = mean_std.float().to(device)
        cid3 = sku_brand_cid3[:, -1].long().to(device)

        model_outputs = model(cid3, decoder_inputs, mean_std, teacher_forcing=True)
        loss, cnt = loss_func(model_outputs.contiguous().view(-1), decoder_targets.contiguous().view(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * cnt
        total_num += cnt

    return epoch_loss / total_num


def evaluate_epoch(model, data_loader, loss_func, decoder_seq_len, device):
    model.eval()

    epoch_loss, total_num = 0, 0
    with torch.no_grad():
        for i, (_, decoder_inputs, decoder_targets, sku_start_points, mean_std, sku_brand_cid3) in enumerate(data_loader):

            decoder_inputs = decoder_inputs.float().to(device)
            decoder_targets = decoder_targets.float().to(device)
            mean_std = mean_std.float().to(device)
            cid3 = sku_brand_cid3[:, -1].long().to(device)

            # turn off teacher forcing
            model_outputs = model(cid3, decoder_inputs, mean_std, decoder_seq_len=decoder_seq_len, teacher_forcing=False)
            loss, cnt = loss_func(model_outputs.contiguous().view(-1), decoder_targets[:, -decoder_seq_len:, :].contiguous().view(-1))
            epoch_loss += loss.item() * cnt
            total_num += cnt

    return epoch_loss / total_num


def MSE_Loss(outputs, targets):
    mse_loss = nn.MSELoss(reduction='elementwise_mean')(outputs, targets)
    return mse_loss, targets.numel()


def save_embedding(target):
    embedding_weights = eval('seq2seq.embedding_' + target + '.weight.data.cpu().numpy()')
    embedding_index = eval('train_set.' + target + '_arr')
    embedding = np.concatenate([embedding_index.reshape((-1, 1)), embedding_weights], axis=1)
    np.save('model/' + target + '_embedding.npy', embedding)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'embedding' in name:
            nn.init.xavier_uniform_(param.data)
        else:
            if len(param.data.size()) == 1:
                nn.init.normal_(param.data, 0, 1)
            else:
                nn.init.xavier_uniform_(param.data)


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
    DATA_PATH = '../data/'
    DECODER_SEQ_LEN = 62
    CID3_SIZE = 10
    EMBEDDING_SIZE_CID3 = 5
    LAYER_NUM = 3
    HEAD_NUM = 6
    KERNEL_SIZE = 2
    PF_HIDDEN_SIZE = 15
    DROPOUT = 0.1
    BATCH_SIZE = 128
    BATCH_SIZE_TEST = 256
    INPUTS_SIZE = 31
    CLIP = 15
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    EPOCH_NUM = 100
    PATIENCE = 10
    WEIGHT_DECAY_RATE = 2

    set_seed()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_set = TimeSeriesDataSet(DATA_PATH, mode='train')
    test_set = TimeSeriesDataSet(DATA_PATH, mode='test')
    # train_sampler = WeightedSampler(DATA_PATH)
    train_sampler = RandomSampler(train_set)
    test_sampler = RandomSampler(test_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_TEST, sampler=test_sampler, num_workers=0)

    seq2seq = Transformer(CID3_SIZE, EMBEDDING_SIZE_CID3,
                          INPUTS_SIZE, LAYER_NUM, HEAD_NUM, KERNEL_SIZE, PF_HIDDEN_SIZE, DROPOUT, device).to(device)

    seq2seq.apply(init_weights)

    optimizer = optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(seq2seq.parameters(), lr=LEARNING_RATE, nesterov=True, momentum=MOMENTUM)

    min_val_loss = float('inf')
    not_descending_cnt = 0
    loss_list = [0.] * PATIENCE
    for epoch in range(EPOCH_NUM):

        start_time = time()

        train_loss = train_epoch(seq2seq, train_loader, optimizer, MSE_Loss, CLIP, device)

        val_loss = evaluate_epoch(seq2seq, test_loader, MSE_Loss, DECODER_SEQ_LEN, device)

        end_time = time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('Epoch: %s | Time: %sm %ss' % (str(epoch + 1).zfill(2), epoch_mins, epoch_secs))
        print('\tTrain Loss: %.3f | Val Loss: %.3f' % (train_loss, val_loss))

        loss_list.pop(0)
        loss_list.append(val_loss)

        if val_loss >= min_val_loss:
            not_descending_cnt += 1
            if not_descending_cnt >= 30 and epoch != EPOCH_NUM - 1:
                print('\nEarly Stopped ...')
                break
            if epoch >= PATIENCE and val_loss >= max(loss_list):
                LEARNING_RATE /= WEIGHT_DECAY_RATE
                print("\n------------  Shrink learning rate to %s\n" % LEARNING_RATE)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LEARNING_RATE
        else:
            not_descending_cnt = 0
            min_val_loss = val_loss
            torch.save(seq2seq.state_dict(), '../model/seq2seq_transformer_model.pt')
            # save_embedding('cid3')
            print()
            print('model saved with validation loss', val_loss)
            print()