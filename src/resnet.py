import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, dim):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(
            nn.GELU(), nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(), nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.layers(x) + x

