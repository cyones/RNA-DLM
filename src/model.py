import torch as tr
import torch.nn as nn
#  from performer_pytorch import Performer
import math as mt
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from src.embedding import NucleotideEmbedding, in_channels
from src.resnet import ResNet
from src.positional_encoding import PositionalEncoding
from src.dataset import MASKED_IDX


class RNADLM(nn.Module):
    def __init__(self, device, class_weights):
        super(RNADLM, self).__init__()
        self.device = device

        self.embedding_dim = 768
        self.embedding = nn.Embedding(
                num_embeddings=4096 + 1,
                embedding_dim=self.embedding_dim,
                padding_idx=0
                )

        self.positional_encodding = PositionalEncoding(self.embedding_dim, max_len=1024)
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=12,
                dim_feedforward=2048,
                activation='gelu'
                ),
            num_layers=10
            )

        self.linear = nn.Linear(self.embedding_dim, 4096)

        self.loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
        #  self.optimizer = tr.optim.Adam(self.parameters())
        self.optimizer = tr.optim.SGD(
            self.parameters(),
            lr=1e-6,
            momentum=0.9,
        )
        self.lr_scheduler = tr.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            scale_mode="exp_range",
            gamma=0.99,
            base_lr=1e-6,
            max_lr=0.5,
            step_size_up=16,
            cycle_momentum=True,
            base_momentum=0.5,
            max_momentum=0.9
        )
        self.optimizer.zero_grad()
        self.to(device = self.device)
        self.requires_grad_(True)

    def forward(self, seq):
        x = seq.to(device = self.device)
        x = self.embedding(x) * mt.sqrt(self.embedding_dim)
        x = self.positional_encodding(x)
        x = x.transpose(0,1)
        x = self.self_attention(x)
        x = x.transpose(0,1)
        x = self.linear(x)
        return x

    def train_step(self, masked_sequence, sequence):
        masked = masked_sequence==MASKED_IDX
        prediction = self(masked_sequence)

        prediction = prediction[masked]
        sequence = sequence[masked]
        loss = self.loss_function(prediction, sequence)
        y_true = sequence.detach().cpu()
        y_pred = prediction.detach().cpu().argmax(dim=1)
        bacc = 100 * balanced_accuracy_score(
                y_true = y_true,
                y_pred = y_pred,
                adjusted = True
                )
        acc = 100 * accuracy_score(
                y_true = y_true,
                y_pred = y_pred
                )
        return loss, acc, bacc

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def load(self, model_file):
        self.load_state_dict(tr.load(model_file, map_location=lambda storage, loc: storage))

    def save(self, model_file):
        tr.save(self.state_dict(), model_file)
