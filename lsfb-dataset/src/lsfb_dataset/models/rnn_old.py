import torch.nn as nn
import torch.jit as jit
from .mogrifier_lstm import MogrifierLSTM
from .edrn_opti import EDRN
from .plstm import PhasedLSTM


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outs, _ = self.lstm(x)
        return self.fc(outs)


class MogrifierLSTMClassifier(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(MogrifierLSTMClassifier, self).__init__()
        self.mogrifier_lstm = MogrifierLSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    @jit.script_method
    def forward(self, x):
        outs, _ = self.mogrifier_lstm(x)
        return self.fc(outs)


class EDRNClassifier(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EDRNClassifier, self).__init__()
        self.edrn = EDRN(input_size, hidden_size, 2)
        self.fc = nn.Linear(hidden_size, num_classes)

    @jit.script_method
    def forward(self, x):
        outs, _ = self.edrn(x)
        return self.fc(outs)


class PLSTMClassifier(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PLSTMClassifier, self).__init__()
        self.plstm = PhasedLSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    @jit.script_method
    def forward(self, x):
        outs, _ = self.plstm(x)
        return self.fc(outs)
