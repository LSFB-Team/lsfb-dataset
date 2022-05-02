import torch
import torch.nn as nn

import math
from typing import Tuple


class EDRN(nn.Module):
    """
    Custom implementation of the Explicit Duration Recurrent Network
    from Shun-Zheng Yu's paper (https://ieeexplore.ieee.org/document/9336287).

    Attributes
    ----------
    input_size : int
        Number N of input features.
    hidden_size : int
        Number M of hidden states.
    num_substates : int
        Number D of substates per hidden state.
    A_th : torch.Tensor
        Interstate transition probability matrix (M, MD) for G_th
    A_ot : torch.Tensor
        Interstate transition probability matrix (M, MD) for G_th
    A_fg_in : torch.Tensor
        Concatenation (M, 2MD) of state transition probability matrices (M, MD) for G_fg and G_in
    A_pt : torch.Tensor
        Sub-state transition probability matrix (MD, MD) for G_th
    A_st : torch.Tensor
        Projection matrix (M, M)
    B : torch.Tensor
        Concatenation (N, 4MD) of the observation probability matrices (N, MD) for G_fg, G_in, G_th and G_ot.
    b : torch.Tensor
        Bias (1, 4MD) for G_fg, G_in, G_th and G_ot.
    """

    def __init__(self, input_size: int, hidden_size: int, num_substates: int):
        """
        Parameters
        ----------
        input_size : int
            Number N of input features.
        hidden_size : int
            Number M of hidden states.
        num_substates : int
            Number D of substates per hidden state.
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_substates = num_substates

        N = input_size
        M = hidden_size
        MD = hidden_size * num_substates

        self.A_th = nn.Parameter(torch.Tensor(M, MD))
        self.A_ot = nn.Parameter(torch.Tensor(M, MD))
        self.A_fg_in = nn.Parameter(torch.Tensor(M, 2 * MD))
        self.A_pt = nn.Parameter(torch.Tensor(MD, MD))
        self.A_st = nn.Parameter(torch.Tensor(M, M))

        self.B = nn.Parameter(torch.Tensor(N, 4 * MD))
        self.b = nn.Parameter(torch.Tensor(1, 4 * MD))

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the parameters.
        """

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            The batch of input sequences that are forwarded to the model.
            The shape of this tensor is assumed (batch_size, sequence_length, D).
            The number D of features must be equals to the input size of the model.

        Returns
        -------
        torch.tensor
            The output batch of sequences.
            The shape of this tensor is assumed (batch_size, sequence_length, hidden_size)

        Tuple[torch.tensor, torch.Tensor]
            The tuple containing the final (h_t, c_t) hidden state of the model.
        """
        batch_size, seq_len, _ = x.size()
        M = self.hidden_size
        MD = self.hidden_size * self.num_substates

        # Initialize the hidden state
        h_t = torch.zeros(batch_size, M).to(x.device)
        c_t = torch.zeros(batch_size, MD).to(x.device)

        hidden_seq = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Optimization: Minimize the number of matrix multiplications
            b_prim = x_t @ self.B + self.b
            a_prim = h_t @ self.A_fg_in + b_prim[:, :MD * 2]

            g_fg = torch.sigmoid(a_prim[:, :MD])
            g_in = torch.sigmoid(a_prim[:, MD:MD * 2])
            g_th = torch.tanh(c_t @ self.A_pt + c_t[:, -M:] @ self.A_th + b_prim[:, MD * 2:MD * 3])
            c_t = c_t * g_fg + g_th * g_in
            g_ot = torch.sigmoid(c_t[:, -M:] @ self.A_ot + b_prim[:, MD * 3:])
            cc_t = torch.tanh(c_t) * g_ot

            h_t = cc_t[:, -M:] @ self.A_st
            for d in range(1, self.num_substates):
                h_t = h_t + cc_t[:, (d - 1) * M:d * M]

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # Hidden Seq: (seq_len, batch_size, M) -> (batch_size, seq_len, M)

        return hidden_seq, (h_t, c_t)


class EDRNClassifier(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            num_substates=2,
    ):
        """
        Parameters
        ----------
        input_size : int
            Number N of input features.
        hidden_size : int
            Number M of hidden states.
        num_classes : int
            Number of classes to predict
        num_substates : int
            Number D of substates per hidden state.
        """
        super(EDRNClassifier, self).__init__()

        self.rnn = EDRN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_substates=num_substates,
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            The batch of input sequences that are forwarded to the model.
            The shape of this tensor is assumed (batch_size, sequence_length, D).
            The number D of features must be equals to the input size of the model.

        Returns
        -------
        torch.tensor
            The prediction of the model.
            The shape of this tensor is assumed (batch_size, sequence_length, num_classes)
        """

        outputs, _ = self.rnn(x)
        return self.fc(outputs)
