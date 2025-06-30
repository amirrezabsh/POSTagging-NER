import torch
from torch import nn
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        self.bias.data.zero_()

    def forward(self, x, h):
        # x: (batch, input_dim), h: (batch, hidden_dim)
        gates = (F.linear(x, self.weight_ih) +
                 F.linear(h, self.weight_hh) +
                 self.bias)
        r, z, n = gates.chunk(3, 1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n + r * F.linear(h, self.weight_hh[self.hidden_dim*2:self.hidden_dim*3]))
        h_new = (1 - z) * n + z * h
        return h_new

class MultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.cells = nn.ModuleList()
        self.cells.append(GRUCell(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.cells.append(GRUCell(hidden_dim, hidden_dim))
        self.linear = nn.Linear(hidden_dim, input_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()

    def forward(self, x, h=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        if h is None:
            h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        else:
            h = [h[i] for i in range(self.num_layers)]

        # Unbind sequence for timestep-wise processing
        x_seq = torch.unbind(x, dim=1)  # List of (batch, input_dim), len = seq_len

        layer_input = x_seq
        new_hidden = []
        for layer_idx, cell in enumerate(self.cells):
            h_t = h[layer_idx]
            outputs = []
            for t in range(seq_len):
                h_t = cell(layer_input[t], h_t)
                outputs.append(h_t)
            # Stack outputs for this layer
            layer_output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            # Prepare input for next layer
            layer_input = torch.unbind(layer_output, dim=1)
            new_hidden.append(h_t.unsqueeze(0))
        # Final output projection
        out = self.linear(self.dropout(layer_output))
        new_hidden = torch.cat(new_hidden, dim=0)  # (num_layers, batch, hidden_dim)
        return out, new_hidden