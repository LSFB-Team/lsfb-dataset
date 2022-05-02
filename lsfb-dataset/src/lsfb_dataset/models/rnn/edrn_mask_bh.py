import torch
from torch import nn, Tensor, sigmoid, tanh
from torch.nn import Parameter, init
import math
from typing import Tuple


class EDRNCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, d: int):
        super(EDRNCell, self).__init__()
        self.hidden_size = hidden_size
        self.d = d

        N = input_size
        M = hidden_size
        MD = M * d

        self.A_fg = Parameter(torch.randn(M, MD))
        self.A_in = Parameter(torch.randn(M, MD))
        self.A_th = Parameter(torch.randn(M, MD))
        self.A_ot = Parameter(torch.randn(M, MD))

        self.A_st = Parameter(torch.randn(M, M))
        self.A_pt = Parameter(torch.randn(MD, MD))

        self.B_fg = Parameter(torch.randn(N, MD))
        self.B_in = Parameter(torch.randn(N, MD))
        self.B_th = Parameter(torch.randn(N, MD))
        self.B_ot = Parameter(torch.randn(N, MD))

        self.b_fg = Parameter(torch.randn(1, MD))
        self.b_in = Parameter(torch.randn(1, MD))
        self.b_th = Parameter(torch.randn(1, MD))
        self.b_ot = Parameter(torch.randn(1, MD))

        self.A_pt_mask = torch.zeros(MD, MD)
        self.A_diag_mask = torch.ones(M, MD)
        for j in range(M):
            self.A_diag_mask.view(M, M, d)[j, j, :] = 0
            for s in range(d):
                self.A_pt_mask.view(M, d, M, d)[j, s, j, s + 1:] = 1
        self.A_pt_mask = Parameter(self.A_pt_mask, requires_grad=False)
        self.A_diag_mask = Parameter(self.A_diag_mask, requires_grad=False)

        self.init_weights()

        self.register_full_backward_hook(self._backward_hook)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        init.uniform_(self.A_fg, -stdv, stdv)
        init.uniform_(self.A_in, -stdv, stdv)
        init.uniform_(self.A_th, -stdv, stdv)
        init.uniform_(self.A_ot, -stdv, stdv)

        init.uniform_(self.A_st, -stdv, stdv)
        init.uniform_(self.A_pt, -stdv, stdv)

        init.uniform_(self.B_fg, -stdv, stdv)
        init.uniform_(self.B_in, -stdv, stdv)
        init.uniform_(self.B_th, -stdv, stdv)
        init.uniform_(self.B_ot, -stdv, stdv)

        init.uniform_(self.b_fg, -stdv, stdv)
        init.uniform_(self.b_in, -stdv, stdv)
        init.uniform_(self.b_th, -stdv, stdv)
        init.uniform_(self.b_ot, -stdv, stdv)

        self.ensure_constraints()

    def ensure_constraints(self):
        self.A_th.data = self.A_th.data * self.A_diag_mask
        self.A_ot.data = self.A_ot.data * self.A_diag_mask
        self.A_pt.data = self.A_pt.data * self.A_pt_mask

    def _backward_hook(self, module, grad_input, grad_output):
        self.ensure_constraints()


    def forward(self, x_t: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        m_t, a_t = state
        B = m_t.size(0)
        M = self.hidden_size
        D = self.d

        G_fg = sigmoid(m_t @ self.A_fg + x_t @ self.B_fg + self.b_fg)
        G_in = sigmoid(m_t @ self.A_in + x_t @ self.B_in + self.b_in)
        G_th = tanh(a_t @ (self.A_pt * self.A_pt_mask) + a_t.view(B, M, D)[:, :, -1] @ (self.A_th * self.A_diag_mask) + x_t @ self.B_th + self.b_th)

        a_t = a_t * G_fg + G_th * G_in

        G_ot = sigmoid(a_t.view(B, M, D)[:, :, -1] @ (self.A_ot * self.A_diag_mask) + x_t @ self.B_ot + self.b_ot)

        aa_t = tanh(a_t) * G_ot
        # aa_t_view = aa_t.view(B, M, D).contiguous()
        m_t = aa_t.view(B, M, D)[:, :, :-1].sum(dim=-1) + aa_t.view(B, M, D)[:, :, -1] @ self.A_st

        return m_t, a_t


class EDRNLayer(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_substates: int):
        super(EDRNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_substates = num_substates
        self.cell = EDRNCell(input_size, hidden_size, d=num_substates)

    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.size()

        m_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        a_t = torch.zeros(batch_size, self.hidden_size * self.num_substates, device=x.device)
        m_t.requires_grad_(False)
        a_t.requires_grad_(False)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :].contiguous()
            m_t, a_t = self.cell(x_t, (m_t, a_t))
            outputs.append(m_t)

        return torch.stack(outputs).transpose(0, 1).contiguous()


class EDRNClassifier(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_substates: int):
        super(EDRNClassifier, self).__init__()

        self.rnn = EDRNLayer(input_size, hidden_size, num_substates)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor):
        out = self.rnn(x)
        return self.fc(out)
