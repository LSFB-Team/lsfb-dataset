import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class LSTM_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_CRF, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size // 2, num_layers=1)
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def init_hidden(self):
        return (
            torch.zeros(2, 1)
        )

