import torch.nn as nn
from .lstm import LSTM
from .edrn import EDRN
from .plstm import PhasedLSTM
from .mogrifier_lstm import MogrifierLSTM


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


class PhasedLSTMClassifier(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            n_classes,
            alpha=1e-3,
            taux_max=3.0,
            r_on=5e-2
    ):
        super(PhasedLSTMClassifier, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes

        self.rnn = PhasedLSTM(
            input_size,
            hidden_size,
            alpha=alpha,
            taux_max=taux_max,
            r_on=r_on,
        )

        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return self.fc(outputs)


class MogrifierLSTMClassifier(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            n_classes,
            mogrify_steps=5,
    ):
        super(MogrifierLSTMClassifier, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes

        self.rnn = MogrifierLSTM(input_size, hidden_size, mogrify_steps=mogrify_steps)

        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return self.fc(outputs)
