import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepARLSTM(nn.Module):
    def __init__(self, individual_size, embedding_size, inputs_size, hidden_size, layer_num, dropout, device):
        super(DeepARLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.device = device

        self.embedding = nn.Embedding(individual_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size+inputs_size, hidden_size, num_layers=layer_num, dropout=dropout)
        self.linear_mu = nn.Linear(hidden_size*layer_num, 1)
        self.linear_sigma = nn.Linear(hidden_size*layer_num, 1)

    def forward(self, individuals, inputs, hidden, cell):
        # individuals: (1, batch)
        # inputs: (1, batch, feature)
        # hidden: (layer, batch, hidden)
        # cell: (layer, batch, hidden)

        individuals_embedded = self.embedding(individuals)
        lstm_inputs = torch.cat([individuals_embedded, inputs], dim=2)
        output, (hidden, cell) = self.lstm(lstm_inputs, (hidden, cell))
        hidden_ = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)
        mu = self.linear_mu(hidden_)
        sigma = F.softplus(self.linear_sigma(hidden_))
        return mu.squeeze(1), sigma.squeeze(1), hidden, cell

    def init_hidden_cell(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        hidden = torch.zeros(self.layer_num, batch_size, self.hidden_size, device=self.device)
        cell = torch.zeros(self.layer_num, batch_size, self.hidden_size, device=self.device)
        return hidden, cell


class DeepARDecoder:
    def __init__(self, input_seq_len, output_seq_len, device):
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.device = device

    def decoder(self, model, individuals, inputs, hidden, cell, means):
        batch_size = inputs.size(1)
        mu_sample = torch.zeros(self.output_seq_len, batch_size, device=self.device)
        sigma_sample = torch.zeros(self.output_seq_len, batch_size, device=self.device)
        for t in range(self.output_seq_len):
            mu, sigma, hidden_, cell_ = model(individuals, inputs[self.input_seq_len+t].unsqueeze(0), hidden, cell)
            mu_sample[t, :] = mu * means
            sigma_sample[t, :] = sigma * means
            if t >= self.output_seq_len - 1:
                break
            inputs[self.input_seq_len+t+1, :, 0] = mu
        return mu_sample, sigma_sample

    def decoder_sampling(self, model, individuals, inputs, hidden, cell, means, sampling_times=100):
        batch_size = inputs.size(1)
        pred_samples = torch.zeros(sampling_times, self.output_seq_len, batch_size, device=self.device)
        for s in range(sampling_times):
            inputs_, hidden_, cell_ = inputs.clone(), hidden.clone(), cell.clone()
            for t in range(self.output_seq_len):
                mu, sigma, hidden_, cell_ = model(individuals, inputs_[self.input_seq_len+t].unsqueeze(0), hidden_, cell_)
                normal_distribution = torch.distributions.normal.Normal(mu, sigma)
                pred_sample = normal_distribution.sample()
                pred_samples[s, t, :] = pred_sample * means
                if t >= self.output_seq_len - 1:
                    break
                inputs_[self.input_seq_len+t+1, :, 0] = pred_sample
        mu_sample = pred_samples.median(dim=0)[0]
        sigma_sample = pred_samples.std(dim=0)
        return pred_samples, mu_sample, sigma_sample


def negative_log_likelihood(mu, sigma, targets):
    nonzero_index = (targets != 0)
    normal_distribution = torch.distributions.normal.Normal(mu[nonzero_index], sigma[nonzero_index])
    log_likelihood = normal_distribution.log_prob(targets[nonzero_index])
    return -log_likelihood.mean()


def mae(mu, targets):
    nonzero_index = (targets != 0)
    return torch.mean((mu[nonzero_index] - targets[nonzero_index]).abs())

def mape(mu, targets):
    nonzero_index = (targets != 0)
    return torch.mean((mu[nonzero_index] - targets[nonzero_index]).abs() / targets[nonzero_index])