import torch as tr
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(SelfAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
            )
        self.normalization = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x + self.normalization(self.multihead_attention(x, x, x)[0])
        return x
