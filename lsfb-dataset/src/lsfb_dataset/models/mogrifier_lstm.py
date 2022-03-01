import torch
import torch.nn as nn


class MogrifierLSTMCell(nn.Module):
    """
    https://github.com/fawazsammani/mogrifier-lstm-pytorch
    """
    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])

        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i+1) % 2 == 0:
                h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x

        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct


class MogrifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MogrifierLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mogrifier_lstm_cell = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps=5)

    def forward(self, x):
        bs, seq_len, _ = x.size()

        h_t = torch.zeros(bs, self.hidden_size).to(x.device)
        c_t = torch.zeros(bs, self.hidden_size).to(x.device)

        hidden_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.mogrifier_lstm_cell(x_t, (h_t, c_t))
            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq)
        hidden_seq = hidden_seq.permute(1, 0, 2)

        return hidden_seq, (h_t, c_t)




