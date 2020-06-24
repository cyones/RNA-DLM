import torch as tr
import torch.nn as nn
from src.embedding import NucleotideEmbedding, in_channels
from src.resnet import ResNet
from src.positional_encoding import PositionalEncoding


class RNADLM(nn.Module):
    def __init__(self, device):
        super(RNADLM, self).__init__()
        self.device = device

        self.embedding = NucleotideEmbedding()

        self.tokenizer = nn.Sequential(
            ResNet(4), ResNet(4), ResNet(4), ResNet(4),
            nn.GELU(), nn.BatchNorm1d(4),
            nn.Conv1d(4, 8, kernel_size=2, stride=2),
            ResNet(8), ResNet(8), ResNet(8), ResNet(8),
            nn.GELU(), nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size=2, stride=2),
            ResNet(16), ResNet(16), ResNet(16), ResNet(16),
            nn.GELU(), nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=2, stride=2),
            ResNet(32), ResNet(32), ResNet(32), ResNet(32),
            nn.GELU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=2, stride=2),
            ResNet(64), ResNet(64), ResNet(64), ResNet(64),
            nn.GELU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=2, stride=2),
            ResNet(128), ResNet(128), ResNet(128), ResNet(128),
            nn.GELU(), nn.BatchNorm1d(128),
        )
        self.self_atention = nn.Sequential(
            PositionalEncoding(128, max_len=1024),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=128, 
                    nhead=4, 
                    dim_feedforward=1024, 
                    dropout=0.1,
                    activation='gelu'
                ),
                num_layers = 8
            )
        )
        self.out_convs = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            ResNet(128), ResNet(128), ResNet(128), ResNet(128),
            ResNet(128), ResNet(128), ResNet(128), ResNet(128),
            ResNet(128), ResNet(128), ResNet(128), ResNet(128),
            ResNet(128), ResNet(128), ResNet(128), ResNet(128),
            nn.Sigmoid()
        )

        self.loss_function = nn.BCELoss()
        self.optimizer = tr.optim.SGD(
            self.parameters(),
            lr=1e-3,
            momentum=0.9,
            weight_decay=1e-5
            )
        self.lr_scheduler = tr.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=1e-3,
            max_lr=2.0,
            step_size_up=2048,
            cycle_momentum=True,
            base_momentum=0.5,
            max_momentum=0.9
            )
        self.to(device = self.device)

    def forward(self, seq):
        seq = seq.to(device = self.device)
        seq = self.embedding(seq)
        seq = self.tokenizer(seq)
        # [sequence length, batch size, embed dim]
        seq = seq.transpose(0,1).transpose(0,2)
        seq = self.self_atention(seq)
        # [batch size, embed dim, sequence length]
        seq = seq.transpose(0,1).transpose(1,2)
        seq = self.out_convs(seq)

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
            seq_pred = pred[i, :, mask_idx[i]]
            seq_targ = masked[i]
            loss += self.loss_function(seq_pred, seq_targ)
            acc += ((seq_pred>0.5) == (seq_targ>0.5)).sum().float() / seq_pred.numel()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        acc = 100 * acc.item() / pred.shape[0]
        return loss.data.item(), acc

    def load(self, model_file):
        self.load_state_dict(tr.load(model_file, map_location=lambda storage, loc: storage))

    def save(self, model_file):
        tr.save(self.state_dict(), model_file)
