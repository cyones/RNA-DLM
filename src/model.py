import torch as tr
import torch.nn as nn
from local_attention import LocalAttention
import math as mt
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from src.positional_encoding import PositionalEncoding
from src.dataset import MASKED_IDX


class RNADLM(nn.Module):
    def __init__(self, device, class_weights):
        super(RNADLM, self).__init__()
        self.device = device

        self.embedding_dim = 768
        self.token_emb = nn.Embedding(
            num_embeddings=4096 + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        self.positional_emb = PositionalEncoding(
            self.embedding_dim,
            max_len=256
        )
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=12,
                dim_feedforward=4096,
                activation='gelu'
            ),
            num_layers=16
        )
        self.linear = nn.Linear(self.embedding_dim, 4096)


        self.loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = tr.optim.SGD(
            self.parameters(),
            lr=1e-6,
            momentum=0.99,
        )
        self.lr_scheduler = tr.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=1e-6,
            max_lr=1e-2,
            step_size_up=32,
            cycle_momentum=True,
            base_momentum=0.9,
            max_momentum=0.99
        )
        self.optimizer.zero_grad()
        self.to(device = self.device)

    def forward(self, seq, mask):
        x = seq.to(device = self.device)
        x = self.token_emb(x) * mt.sqrt(self.embedding_dim)
        x = self.positional_emb(x)
        x = x.transpose(0,1)
        if mask is not None:
            mask = mask.to(device = self.device)
            x = self.self_attention(x, src_key_padding_mask=mask)
        else:
            x = self.self_attention(x)
        x = x.transpose(0,1)
        x = self.linear(x)
        return x

    def train_step(self, sequence, mask):
        sequence = sequence.to(device = self.device)
        y_true = sequence[~mask]

        sequence[~mask] = 0
        y_pred = self(sequence, mask)[~mask]

        loss = self.loss_function(y_pred, y_true)
        y_pred = y_pred.detach().cpu().argmax(dim=1)
        y_true = y_true.detach().cpu()
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
