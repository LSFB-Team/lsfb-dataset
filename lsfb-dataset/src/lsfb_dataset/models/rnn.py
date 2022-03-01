import torch.nn as nn
from .mogrifier_lstm import MogrifierLSTM


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outs, _ = self.lstm(x)
        return self.fc(outs)


class MogrifierLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(MogrifierLSTMClassifier, self).__init__()
        self.mogrifier_lstm = MogrifierLSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outs, _ = self.mogrifier_lstm(x)
        return self.fc(outs)
