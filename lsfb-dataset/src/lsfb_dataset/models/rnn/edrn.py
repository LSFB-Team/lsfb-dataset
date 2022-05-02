import torch
from torch import nn, jit, Tensor
from torch.nn import Parameter, init
import math
from typing import Tuple, List


class EDRNCell(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, num_substates: int):
        super(EDRNCell, self).__init__()

        self.hidden_size = hidden_size
        self.num_substates = num_substates

        N = input_size
        M = hidden_size
        MD = hidden_size * num_substates

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

    def init_weights(self, hidden_size: int):
        stdv = 1.0 / math.sqrt(hidden_size)
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

    @jit.script_method
    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        mx, ax = state

        BS, _ = x.size()

        M = self.hidden_size
        D = self.num_substates

        G_fg = mx @ self.A_fg + x @ self.B_fg + self.b_fg
        G_fg = torch.sigmoid(G_fg)

        G_in = mx @ self.A_in + x @ self.B_in + self.b_in
        G_in = torch.sigmoid(G_in)

        G_th = ax @ self.A_pt + ax.view(BS, M, D)[:, :, -1] @ self.A_th + x @ self.B_th + self.b_th
        G_th = torch.tanh(G_th)

        ax = ax * G_fg + G_th * G_in

        G_ot = ax.view(BS, M, D)[:, :, -1] @ self.A_ot + x @ self.B_ot + self.b_ot
        G_ot = torch.sigmoid(G_ot)

        ax_star = torch.tanh(ax) * G_ot

        mx = torch.sum(ax_star.view(BS, M, D), dim=-1) + ax_star.view(BS, M, D)[:, :, -1] @ self.A_st

        return mx, (mx, ax)


class EDRNLayer(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, num_substates: int):
        super(EDRNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_substates = num_substates
        self.cell = EDRNCell(input_size, hidden_size, num_substates)

    @jit.script_method
    def init_state(self, x: Tensor):
        hx = x.new_zeros(x.size(0), self.hidden_size).to(x.device)
        cx = x.new_zeros(x.size(0), self.hidden_size * self.num_substates).to(x.device)
        hx.requires_grad_(False)
        cx.requires_grad_(False)
        return hx, cx

    @jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # x: (batch_size, seq_len, D)

        state = self.init_state(x)
        outputs = torch.jit.annotate(List[Tensor], [])
        bs, seq_len, _ = x.size()

        for t in range(0, seq_len):
            # x_t: (batch_size, D)
            x_t = x[:, t, :]
            out, state = self.cell(x_t, state)
            outputs.append(out)

        outputs = torch.stack(outputs).transpose(1, 0)
        # outputs: (seq_len, batch_size, D) -> (batch_size, seq_len, D)
        return outputs, state


class EDRNClassifier(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_substates: int):
        super(EDRNClassifier, self).__init__()
        self.edrn = EDRNLayer(input_size, hidden_size, num_substates)
        self.fc = nn.Linear(hidden_size, num_classes)

    @jit.script_method
    def forward(self, x: Tensor):
        x, _ = self.edrn(x)
        return self.fc(x)

