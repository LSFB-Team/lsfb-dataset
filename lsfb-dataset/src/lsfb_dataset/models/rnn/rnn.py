import torch
from torch import nn


_lstm_modes = ['LSTM', 'EDRN', 'mLSTM']


class LSTMBase(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            bidirectional=False,
            num_layers=1,
            **kwargs
    ):
        super(LSTMBase, self).__init__()





