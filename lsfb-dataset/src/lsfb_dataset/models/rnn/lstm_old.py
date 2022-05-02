import torch
import torch.nn as nn
import torch.jit as jit
from ..drop.variational_dropout import VariationalDropout


class LSTM(nn.LSTM):
    """
    Better LSTM.
    See https://arxiv.org/abs/1512.05287
    Code from https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
    """
    def __init__(
            self,
            *args,
            dropout_in: float = 0.,
            dropout_w: float = 0.,
            dropout_out: float = 0.,
            unit_forget_bias=True,
            **kwargs
    ):
        super(LSTM, self).__init__(*args, **kwargs, batch_first=True)

        self.unit_forget_bias = unit_forget_bias
        self.dropout_w = dropout_w
        self.input_drop = VariationalDropout(dropout_in, batch_first=True)
        self.output_drop = VariationalDropout(dropout_out, batch_first=True)

        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1
        self.flatten_parameters()

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                # noinspection PyUnresolvedReferences
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropout_w, training=self.training).contiguous()
        self.flatten_parameters()

    def forward(self, x, hx=None):
        self._drop_weights()
        x = self.input_drop(x)
        seq, state = super().forward(x, hx=hx)
        return self.output_drop(seq), state



