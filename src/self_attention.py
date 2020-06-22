import torch as tr
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout)
        super(SelfAttention, self).__init__()
        self.multihead_attention = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
            )
        self.normalization = nn.LayerNorm()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = x + self.multihead_attention(x)
        x = self.normalization(x)
        x = x.transpose(1, 2)
        return x
