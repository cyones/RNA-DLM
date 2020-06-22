import torch as tr
import torch.nn as nn

in_channels = 4

class NucleotideEmbedding(nn.Module):
    def __init__(self):
        super(NucleotideEmbedding, self).__init__()
        weight = tr.Tensor([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 1],
                            [0, 1, 1, 0],
                            [0, 0, 1, 1],
                            [1, 1, 0, 0],
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 1, 1],
                            [1, 0, 1, 1],
                            [1, 1, 0, 1],
                            [1, 1, 1, 0],
                            [1, 1, 1, 1]])
        self.embedding = nn.Embedding.from_pretrained(weight)

    def forward(self, x):
        x = x.to(tr.int64)
        x = self.embedding(x)
        x = x.transpose(1,2)
        return x

