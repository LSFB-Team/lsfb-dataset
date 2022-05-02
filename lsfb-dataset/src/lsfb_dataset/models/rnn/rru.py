import torch
from torch import nn, Tensor
from torch.nn import Parameter


class RRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, relu_layers=1):
        super(RRUCell, self).__init__()

        w_x = Parameter(torch.randn(input_size, hidden_size))
        w_h = Parameter(torch.randn(hidden_size, hidden_size))
        b_j = Parameter(torch.randn(hidden_size))


