
import torch
from torch import jit, nn, Tensor
from torch.nn import init, Parameter
from typing import List, Tuple
import math


class PLSTMCell(jit.ScriptModule):
    """
    Reference
        https://github.com/Marcovaldong/lstmp.pytorch
    """

    def __init__(self, input_size: int, hidden_size: int, projection_size: int):
        super(PLSTMCell, self).__init__()

        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, projection_size))
        self.weight_hr = Parameter(torch.randn(projection_size, hidden_size))

        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

        self.init_weights(hidden_size)

    def init_weights(self, hidden_size: int):
        stdv = 1.0 / math.sqrt(hidden_size)
        init.uniform_(self.weight_ih, -stdv, stdv)
        init.uniform_(self.weight_hh, -stdv, stdv)
        init.uniform_(self.weight_hr, -stdv, stdv)
        init.uniform_(self.bias_ih)
        init.uniform_(self.bias_hh)

    @jit.script_method
    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state

        gates = (
            torch.mm(x, self.weight_ih.t()) + self.bias_ih +
            torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        )
        forget_gate, in_gate, cell_gate, out_gate = gates.chunk(4, 1)

        forget_gate = torch.sigmoid(forget_gate)
        in_gate = torch.sigmoid(in_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        hy = out_gate * torch.tanh(cy)
        hy = torch.mm(hy, self.weight_hr.t())

        return hy, (hy, cy)


class PLSTMLayer(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, projection_size: int):
        super(PLSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.cell = PLSTMCell(input_size, hidden_size, projection_size)

    @jit.script_method
    def init_state(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        hx = x.new_zeros(x.size(1), self.projection_size)
        cx = x.new_zeros(x.size(1), self.hidden_size)
        hx.requires_grad_(False)
        cx.requires_grad_(False)
        return hx, cx

    @jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        state = self.init_state(x)
        inputs = x.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for x_t in inputs:
            out, state = self.cell(x_t, state)
            outputs.append(out)
        return torch.stack(outputs), state


class PLSTMClassifier(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, projection_size: int, num_classes: int):
        super(PLSTMClassifier, self).__init__()
        self.plstm = PLSTMLayer(input_size, hidden_size, projection_size)
        self.fc = nn.Linear(projection_size, num_classes)

    @jit.script_method
    def forward(self, x: Tensor):
        x, _ = self.plstm(x)
        return self.fc(x)
