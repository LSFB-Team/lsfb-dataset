import torch
import torch.jit as jit
import torch.nn as nn
import math


class PhasedLSTM(jit.ScriptModule):
    def __init__(self,
                 input_size,
                 hidden_size,
                 alpha=1e-3,
                 taux_max=3.0,
                 r_on=5e-2):
        super(PhasedLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        stdv: float = 1.0 / math.sqrt(self.hidden_size)

        self.W = nn.Parameter(
            torch.FloatTensor(4 * input_size, 4 * hidden_size).uniform_(-stdv, stdv)
        )
        self.U = nn.Parameter(
            torch.FloatTensor(4 * hidden_size, 4 * hidden_size).uniform_(-stdv, stdv)
        )
        self.b = nn.Parameter(
            torch.FloatTensor(4 * hidden_size).zero_()
        )

        self.w_peep = nn.Parameter(
            torch.FloatTensor(3 * hidden_size).uniform_(-stdv, stdv)
        )

        # ---- PLSTM
        self.alpha = alpha
        self.r_on = r_on

        self.tau = nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(0, taux_max).exp_()
        )

        self.shift = nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(0, torch.mean(self.tau).item())
        )

    @jit.script_method
    def fmod(self, a, b):
        return (b / math.pi) * torch.arctan(torch.tan(math.pi * (a / b - 0.5))) + b / 2

    @jit.script_method
    def forward(self, x):
        hs = self.hidden_size
        n = self.input_size

        h = torch.zeros((x.size(0), hs)).to(x.device)
        c = torch.zeros((x.size(0), hs)).to(x.device)

        h_out = []
        x_seq = x.permute(1, 0, 2)
        # x: (B, seq, F) --> (seq, B, F)

        seq_len = x_seq.size(0)
        times = torch.arange(seq_len).unsqueeze(dim=1)
        # times: (seq, 1)
        times = times.expand((seq_len, hs)).to(x.device)
        # times: (seq, hs)
        phi = self.fmod((times - self.shift), self.tau) / (self.tau + 1e-8)

        alpha = self.alpha
        if not self.training:
            alpha = 0.0

        k = torch.where(
            phi < 0.5 * self.r_on,
            2.0 * phi / self.r_on,
            torch.where(
                torch.logical_and(0.5 * self.r_on <= phi, phi < self.r_on),
                2.0 - (2.0 * phi / self.r_on),
                alpha * phi
            )
        )

        for t, x_t in enumerate(x_seq):
            gates = x_t.repeat(1, 3) @ self.W[:n * 3, :hs * 3] + \
                    h.repeat(1, 3) @ self.U[:hs * 3, :hs * 3] + \
                    self.b[:hs * 3] + \
                    self.w_peep * c.repeat(1, 3)

            i_t = torch.sigmoid(gates[:, 0:hs])
            f_t = torch.sigmoid(gates[:, hs:hs * 2])
            o_t = torch.sigmoid(gates[:, hs * 2:hs * 3])

            gate_c = \
                x_t @ self.W[n * 3:n * 4, hs * 3:hs * 4] + \
                h @ self.U[hs * 3:hs * 4, hs * 3:hs * 4] + \
                self.b[hs * 3:hs * 4]

            c_prim = f_t * c + i_t * torch.tanh(gate_c)
            c = k[t] * c_prim + (1 - k[t]) * c
            h_prim = torch.tanh(c_prim) * o_t
            h = k[t] * h_prim + (1 - k[t]) * h
            h_out.append(h)

        t_h_out = torch.stack(h_out)
        t_h_out = t_h_out.permute(1, 0, 2)
        # (seq, B, F) --> (B, seq, F)

        return t_h_out, (h, c)
