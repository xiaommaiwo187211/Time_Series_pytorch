import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

from seq2seq_mqrnn.dataloader import TimeSeriesDataSet
from seq2seq_mqrnn.seq2seq_model import EncoderCho, DecoderMLP, Seq2SeqMQRNN

import numpy as np
from time import time
from functools import partial




def train_epoch(model, data_loader, optimizer, loss_func, clip, device):
    model.train()

    epoch_loss, total_num = 0, 0
    for i, (_, encoder_inputs, decoder_inputs, decoder_targets, sku_start_points, sku_brand_cid3) in enumerate(data_loader):
        # encoder_inputs: (encoder_sequence, batch, feature)
        # decoder_inputs: (decoder_sequence, batch, feature)
        # decoder_targets: (decoder_sequence, batch, 1)
        # sku_brand_cid3: (batch, 3)

        optimizer.zero_grad()

        encoder_inputs = encoder_inputs.float().to(device).permute(1, 0, 2)
        decoder_inputs = decoder_inputs.float().to(device).permute(1, 0, 2)
        decoder_targets = decoder_targets.float().to(device).permute(1, 0, 2)
        cid3 = sku_brand_cid3[:, -1].long().to(device)

        model_outputs = model(cid3, encoder_inputs, decoder_inputs)
        loss, cnt = loss_func(model_outputs, decoder_targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * cnt
        total_num += cnt

    return epoch_loss / total_num


def evaluate_epoch(model, data_loader, loss_func, device):
    model.eval()

    epoch_loss, total_num = 0, 0
    with torch.no_grad():
        for i, (_, encoder_inputs, decoder_inputs, decoder_targets, sku_start_points, sku_brand_cid3) in enumerate(data_loader):
            encoder_inputs = encoder_inputs.float().to(device).permute(1, 0, 2)
            decoder_inputs = decoder_inputs.float().to(device).permute(1, 0, 2)
            decoder_targets = decoder_targets.float().to(device).permute(1, 0, 2)
            cid3 = sku_brand_cid3[:, -1].long().to(device)

            model_outputs = model(cid3, encoder_inputs, decoder_inputs)
            loss, cnt = loss_func(model_outputs, decoder_targets)
            epoch_loss += loss.item() * cnt
            total_num += cnt

    return epoch_loss / total_num


def Quantile_Loss(ypred, ytrue, quantile_list):
    # ytrue: (decoder_sequence, batch, 1)
    # ypred: (decoder_sequence, batch, quantile)

    ytrue = ytrue.contiguous().view(-1)
    loss = 0
    for i, q in enumerate(quantile_list):
        ytrue_, ypred_ = ytrue.clone(), ypred.clone()
        ypred_q = ypred_[:, :, i].contiguous().view(-1)
        notnan_index = (torch.isnan(ypred_q) == 0)
        ypred_q, ytrue_ = ypred_q[notnan_index], ytrue_[notnan_index]
        zeros = torch.zeros_like(ytrue_)
        diff = ytrue_ - ypred_q
        loss_q = 2 * (q * torch.max(diff, zeros) + (1 - q) * torch.max(-diff, zeros))
        loss += torch.sum(loss_q)

    cnt = ypred.numel()
    return loss / cnt, cnt


def save_embedding(target):
    embedding_weights = eval('seq2seq.embedding_' + target + '.weight.data.cpu().numpy()')
    embedding_index = eval('train_set.' + target + '_arr')
    embedding = np.concatenate([embedding_index.reshape((-1, 1)), embedding_weights], axis=1)
    np.save('model/' + target + '_embedding.npy', embedding)


def init_weights(m):
    for name, param in m.named_parameters():
        if len(param.data.size()) == 1:
            nn.init.normal_(param.data, 0, 1)
        else:
            nn.init.xavier_normal_(param.data)


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
    BATCH_SIZE = 256
    BATCH_SIZE_TEST = 256
    ENCODER_INPUTS_SIZE = 33
    DECODER_INPUTS_SIZE = 29
    ENCODER_HIDDEN_SIZE = 32
    DECODER_HIDDEN_SIZE = 32
    CID3_SIZE = 10
    EMBEDDING_SIZE_CID3 = 5
    LAYER_NUM = 1
    DROPOUT = 0.6
    CLIP = 15
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    EPOCH_NUM = 100
    PATIENCE = 10
    WEIGHT_DECAY_RATE = 2

    DECODER_SEQ_LEN = 62
    QUANTILE_LIST = [0.1, 0.5, 0.9]
    OUTPUT_SIZE = len(QUANTILE_LIST)

    set_seed()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_set = TimeSeriesDataSet(DATA_PATH, mode='train')
    test_set = TimeSeriesDataSet(DATA_PATH, mode='test')
    # train_sampler = WeightedSampler(DATA_PATH)
    train_sampler = RandomSampler(train_set)
    test_sampler = RandomSampler(test_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_TEST, sampler=test_sampler, num_workers=0)

    encoder = EncoderCho(EMBEDDING_SIZE_CID3, ENCODER_INPUTS_SIZE, ENCODER_HIDDEN_SIZE, LAYER_NUM)
    decoder = DecoderMLP(EMBEDDING_SIZE_CID3, DECODER_INPUTS_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, DECODER_SEQ_LEN, OUTPUT_SIZE)
    seq2seq = Seq2SeqMQRNN(encoder, decoder, CID3_SIZE, EMBEDDING_SIZE_CID3, DROPOUT, device).to(device)

    seq2seq.apply(init_weights)

    optimizer = optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    # optimizer = optim.SGD(seq2seq.parameters(), lr=LEARNING_RATE, nesterov=True, momentum=MOMENTUM)
    quantile_loss = partial(Quantile_Loss, quantile_list=QUANTILE_LIST)

    min_val_loss = float('inf')
    not_descending_cnt = 0
    loss_list = [0.] * PATIENCE
    for epoch in range(EPOCH_NUM):

        start_time = time()

        train_loss = train_epoch(seq2seq, train_loader, optimizer, quantile_loss, CLIP, device)

        val_loss = evaluate_epoch(seq2seq, test_loader, quantile_loss, device)

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
            torch.save(seq2seq.state_dict(), '../model/seq2seq_mqrnn_model.pt')
            # save_embedding('cid3')
            print()
            print('model saved with validation loss', val_loss)
            print()
