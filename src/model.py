import torch as tr
import torch.nn as nn
import math as mt
import numpy as np
from src.embedding import NucleotideEmbedding, in_channels
from src.resnet import ResNet


class RNADLM(nn.Module):
    def __init__(self, device):
        super(RNADLM, self).__init__()
        self.device = device

        self.embedding = NucleotideEmbedding()

        self.convolutions = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=32, stride=32),
            ResNet(128), ResNet(128), ResNet(128), ResNet(128),
            ResNet(128), ResNet(128), ResNet(128), ResNet(128),
            nn.ELU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.loss_function = nn.MSELoss()
        self.optimizer = tr.optim.Adam(self.parameters(), lr=5e-3, weight_decay=1e-5)
        self.lr_scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5,
            patience=100, min_lr=1e-6, eps=1e-9
            )

        self.to(device = self.device)

    def forward(self, seq):
        seq = seq.to(device = self.device)
        seq = self.embedding(seq)
        seq = self.convolutions(seq)
        seq = seq.transpose(1,2).\
            reshape(-1, int(seq.shape[2]*32), int(seq.shape[1]/32)).\
            transpose(1,2)
        return seq

    def train_step(self, seq, mask_idx, masked):
        self.optimizer.zero_grad()
        pred = self(seq)
        masked = self.embedding(masked)
        
        loss = 0
        acc = 0
        for i in range(pred.shape[0]):
            loss += self.loss_function(pred[i, :, mask_idx[i]], masked[i])
            import ipdb; ipdb.set_trace()
        loss.backward()
        self.optimizer.step()
        return loss.data.item()

    def load(self, model_file):
        self.load_state_dict(tr.load(model_file, map_location=lambda storage, loc: storage))

    def save(self, model_file):
        tr.save(self.state_dict(), model_file)
