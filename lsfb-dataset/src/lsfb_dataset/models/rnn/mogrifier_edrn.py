import torch
import torch.nn as nn
from torch import Tensor

import math


class MogrifierEDRN(nn.Module):
    def __init__(self, input_size, hidden_size, substates=2, mogrify_steps=5):
        super(MogrifierEDRN, self).__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.substates = substates

        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])
        self.mogrify_steps = mogrify_steps

        N = input_size
        M = hidden_size
        MD = hidden_size * substates

        self.A_th = nn.Parameter(torch.Tensor(M, MD))
        self.A_ot = nn.Parameter(torch.Tensor(M, MD))

        self.A_fg_in = nn.Parameter(torch.Tensor(M, 2 * MD))

        self.A_pt = nn.Parameter(torch.Tensor(MD, MD))

        self.A_st = nn.Parameter(torch.Tensor(M, M))

        self.B = nn.Parameter(torch.Tensor(N, 4 * MD))

        self.b = nn.Parameter(torch.Tensor(1, 4 * MD))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def mogrify(self, x: Tensor, h: Tensor):
        for index, mogrifier in enumerate(self.mogrifier_list):
            if index % 2 == 0:
                x = (2 * torch.sigmoid(mogrifier(h))) * x
            else:
                h = (2 * torch.sigmoid(mogrifier(x))) * h

        return x, h

    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_len, _ = x.size()
        M = self.hidden_size
        MD = self.hidden_size * self.substates

        h_t = torch.zeros(batch_size, M).to(x.device)
        c_t = torch.zeros(batch_size, MD).to(x.device)

        hidden_seq = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            x_t, h_t = self.mogrify(x_t, h_t)

            b_prim = x_t @ self.B + self.b
            a_prim = h_t @ self.A_fg_in + b_prim[:, :MD * 2]

            g_fg = torch.sigmoid(a_prim[:, :MD])
            g_in = torch.sigmoid(a_prim[:, MD:MD * 2])
            g_th = torch.tanh(c_t @ self.A_pt + c_t[:, -M:] @ self.A_th + b_prim[:, MD * 2:MD * 3])
            c_t = c_t * g_fg + g_th * g_in
            g_ot = torch.sigmoid(c_t[:, -M:] @ self.A_ot + b_prim[:, MD * 3:])
            cc_t = torch.tanh(c_t) * g_ot

            h_t = cc_t[:, -M:] @ self.A_st
            for d in range(1, self.substates):
                h_t = h_t + cc_t[:, (d - 1) * M:d * M]

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # Hidden Seq: (seq_len, batch_size, d) -> (batch_size, seq_len, d)

        return hidden_seq, (h_t, c_t)
