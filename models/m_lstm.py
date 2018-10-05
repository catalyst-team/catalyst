import torch
import torch.nn as nn
import torch.nn.functional as F


class mLSTM(nn.Module):
    """
    source: https://github.com/guillitte/pytorch-sentiment-neuron
    """

    def __init__(self, data_size, hidden_size):
        super(mLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.data_size = data_size

        self.wx = nn.Linear(data_size, 4 * hidden_size, bias=False)
        self.wh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.wmx = nn.Linear(data_size, hidden_size, bias=False)
        self.wmh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, data, last_hidden):
        hx, cx = last_hidden
        m = self.wmx(data) * self.wmh(hx)
        gates = self.wx(data) + self.wh(m)
        i, f, o, u = gates.chunk(4, 2)

        i = F.sigmoid(i)
        f = F.sigmoid(f)
        u = F.tanh(u)
        o = F.sigmoid(o)

        cy = f * cx + i * u
        hy = o * F.tanh(cy)

        return hy, cy


class StackedLSTM(nn.Module):
    def __init__(
            self, cell, num_layers,
            input_size, hidden_size, output_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.rnn_size = hidden_size
        self.h2o = nn.Linear(hidden_size, output_size)

        self.layers = []
        for i in range(num_layers):
            layer = cell(input_size, hidden_size)
            self.add_module("layer_%d" % i, layer)
            self.layers += [layer]
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            if i == 0:
                input = h_1_i
            else:
                input = input + h_1_i
            if i != len(self.layers):
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        output = self.h2o(input)

        return output, (h_1, c_1)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))
