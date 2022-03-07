import torch.nn as nn
from .lstm import LSTM
from .edrn import EDRN


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            n_classes,
            n_layers=1,
            dropout_in=0.0,
            dropout_w=0.0,
            dropout_out=0.0,
    ):
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes

        if dropout_in or dropout_w or dropout_out:
            self.rnn = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout_in=dropout_in,
                dropout_w=dropout_w,
                dropout_out=dropout_out,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
            )

        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return self.fc(outputs)


class EDRNClassifier(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            n_classes,
            substates=2,
    ):
        super(EDRNClassifier, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes

        self.rnn = EDRN(
            input_size=input_size,
            hidden_size=hidden_size,
            substates=substates,
        )

        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return self.fc(outputs)
