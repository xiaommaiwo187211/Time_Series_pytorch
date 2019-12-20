import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

from electricity_tcn.data_loader import TimeSeriesDataSet
from electricity_tcn.tcn_model import EncoderTCN, DecoderMLP, Seq2SeqDeepTCN

import numpy as np
from time import time
from functools import partial




def train_epoch(model, data_loader, optimizer, loss_func10, loss_func50, loss_func90, device):
    model.train()

    epoch_loss, total_num = 0, 0
    for i, (_, sub_X, sub_Y, future_Y) in enumerate(data_loader):

        optimizer.zero_grad()

        sub_X = sub_X.float().to(device)
        sub_Y = sub_Y.float().to(device)
        future_Y = future_Y.float().to(device)

        model_output10, model_output50, model_output90 = model(sub_X, future_Y)
        loss10 = loss_func10(model_output10, sub_Y)
        loss50 = loss_func50(model_output50, sub_Y)
        loss90 = loss_func90(model_output90, sub_Y)
        loss = loss10 + loss50 + loss90
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss


def evaluate_epoch(model, data_loader, loss_func10, loss_func50, loss_func90, device):
    model.eval()

    pred50_mat = torch.zeros(len(data_loader.dataset), 24, device=device)
    target_mat = torch.zeros(len(data_loader.dataset), 24, device=device)
    epoch_loss, total_num = 0, 0
    with torch.no_grad():
        for i, (_, sub_X, sub_Y, future_Y) in enumerate(data_loader):
            sub_X = sub_X.float().to(device)
            sub_Y = sub_Y.float().to(device)
            future_Y = future_Y.float().to(device)
            batch_size = future_Y.size(0)

            model_output10, model_output50, model_output90 = model(sub_X, future_Y)
            loss10 = loss_func10(model_output10, sub_Y)
            loss50 = loss_func50(model_output50, sub_Y)
            loss90 = loss_func90(model_output90, sub_Y)
            loss = loss10 + loss50 + loss90
            epoch_loss += loss.item()

            pred50_mat[i*batch_size:(i+1)*batch_size] = model_output50
            target_mat[i*batch_size:(i+1)*batch_size] = sub_Y

    pred50_mat, target_mat = pred50_mat.cpu().numpy(), target_mat.cpu().numpy()
    val_ND = ND(pred50_mat, target_mat)
    val_NRMSE = NRMSE(pred50_mat, target_mat)

    return epoch_loss, val_ND, val_NRMSE


class QuantileLoss(nn.Module):
    def __init__(self, q=0.5):
        super(QuantileLoss, self).__init__()
        self.q = q

    def forward(self, pred, target):
        target = target.view(pred.shape)
        I = (pred <= target).float()
        loss = self.q * (target - pred) * I + (1 - self.q) * (pred - target) * (1 - I)
        return torch.sum(loss)


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


def ND(y_pred, y_true):
    demoninator = np.sum(np.abs(y_true))
    diff = np.sum(np.abs(y_true - y_pred))
    return 1.0 * diff / demoninator


def NRMSE(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    denominator = np.mean(y_true)
    diff = np.sqrt(np.mean(((y_pred-y_true)**2)))
    return diff/denominator



if __name__ == '__main__':
    DATA_PATH = 'data/feature_prepare.pkl'
    BATCH_SIZE = 512
    BATCH_SIZE_TEST = 512
    EPOCH_NUM = 500
    LEARNING_RATE = 0.5

    set_seed()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    train_set = TimeSeriesDataSet(DATA_PATH, mode='train')
    test_set = TimeSeriesDataSet(DATA_PATH, mode='test')
    train_sampler = RandomSampler(train_set)
    test_sampler = RandomSampler(test_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_TEST, sampler=test_sampler, num_workers=0)

    seq2seq = Seq2SeqDeepTCN().to(device)

    seq2seq.apply(init_weights)

    optimizer = optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)
    loss_func10 = QuantileLoss(q=0.1)
    loss_func50 = QuantileLoss(q=0.5)
    loss_func90 = QuantileLoss(q=0.9)

    min_val_loss = float('inf')
    not_descending_cnt = 0
    for epoch in range(EPOCH_NUM):

        start_time = time()

        train_loss = train_epoch(seq2seq, train_loader, optimizer, loss_func10, loss_func50, loss_func90, device)

        val_loss, val_ND, val_NRMSE = evaluate_epoch(seq2seq, test_loader, loss_func10, loss_func50, loss_func90, device)

        end_time = time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('Epoch: %s | Time: %sm %ss' % (str(epoch + 1).zfill(2), epoch_mins, epoch_secs))
        print('\tTrain Loss: %.3f | Val Loss: %.3f' % (train_loss, val_loss))
        print('\tVal ND: %.3f | Val NRMSE: %.3f' % (val_ND, val_NRMSE))

        if val_loss >= min_val_loss:
            not_descending_cnt += 1
            if not_descending_cnt >= 20 and epoch >= 19 and epoch != EPOCH_NUM - 1:
                print('Early Stopped ...')
                break
        else:
            not_descending_cnt = 0
            if epoch >= 2:
                min_val_loss = val_loss
                torch.save(seq2seq.state_dict(), 'model/seq2seq_tcn_model.pt')
                print()
                print('model saved with validation loss', val_loss)
                print()
