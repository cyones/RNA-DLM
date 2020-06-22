import torch as tr
import math as mt

class PositionalEncoding(tr.nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = tr.zeros(max_len, d_model)
        position = tr.arange(0, max_len, dtype=tr.float).unsqueeze(1)
        div_term = tr.exp(tr.arange(0, d_model, 2).float() * (-mt.log(10000.0) / d_model))
        pe[:, 0::2] = tr.sin(position * div_term)
        pe[:, 1::2] = tr.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1,2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:,:x.size(2)]
        return x