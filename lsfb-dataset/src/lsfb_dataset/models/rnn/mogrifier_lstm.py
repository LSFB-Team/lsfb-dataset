import torch
import torch.nn as nn
import torch.jit as jit

from torch import Tensor
from typing import Tuple


class MogrifierLSTMCell(jit.ScriptModule):
    """
    https://github.com/fawazsammani/mogrifier-lstm-pytorch
    """

    def __init__(self, input_size, hidden_size, mogrify_steps=5):
        super(MogrifierLSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])
        self.mogrify_steps = mogrify_steps

        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])

    @jit.script_method
    def mogrify(self, x: Tensor, h: Tensor):

        for index, mogrifier in enumerate(self.mogrifier_list):
            if index % 2 == 0:
                x = (2 * torch.sigmoid(mogrifier(h))) * x
            else:
                h = (2 * torch.sigmoid(mogrifier(x))) * h

        return x, h

    @jit.script_method
    def forward(self, x: Tensor, states: Tuple[Tensor, Tensor]):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct


class MogrifierLSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, mogrify_steps=5):
        super(MogrifierLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mogrifier_lstm_cell = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps=mogrify_steps)

    @jit.script_method
    def forward(self, x: Tensor):
        bs, seq_len, _ = x.size()

        h_t = torch.zeros(bs, self.hidden_size).to(x.device)
        c_t = torch.zeros(bs, self.hidden_size).to(x.device)

        hidden_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :].contiguous()
            h_t, c_t = self.mogrifier_lstm_cell(x_t, (h_t, c_t))
            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq)
        hidden_seq = hidden_seq.permute(1, 0, 2)

        return hidden_seq, (h_t, c_t)
