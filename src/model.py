import torch as tr
import torch.nn as nn
#  from performer_pytorch import Performer
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from src.embedding import NucleotideEmbedding, in_channels
from src.resnet import ResNet
from src.positional_encoding import PositionalEncoding


class RNADLM(nn.Module):
    def __init__(self, device, class_weights):
        super(RNADLM, self).__init__()
        self.device = device

        embedding_dim = 768
        self.embedding = nn.Embedding(
                num_embeddings=4096 + 2,
                embedding_dim=embedding_dim,
                padding_idx=0
                )

        self.self_atention = nn.Sequential(
            PositionalEncoding(embedding_dim, max_len=1024),
            #  Performer(
                #  dim = embedding_dim,
                #  local_attn_heads = 1,
                #  depth = 6,
                #  heads = 8,
                #  causal = False,
                #  kernel_fn = nn.GELU(),
                #  ff_dropout = 0.1,
                #  attn_dropout = 0.1
            #  )
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=12,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation='gelu'
                ),
                num_layers = 16
            )
        )

        self.linear = nn.Linear(embedding_dim, 4096)

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
            gamma=0.95,
            base_lr=1e-6,
            max_lr=0.5,
            step_size_up=16,
            cycle_momentum=True,
            base_momentum=0.5,
            max_momentum=0.9
        )
        self.to(device = self.device)

    def forward(self, seq):
        x = seq.to(device = self.device)
        x = self.embedding(x)
        x = self.self_atention(x)
        x = self.linear(x).transpose(1,2)
        return x

    def train_step(self, masked_sequence, sequence):
        masked = masked_sequence==4097
        prediction = self(masked_sequence)

        prediction = prediction.transpose(1,2)[masked]
        sequence = sequence[masked] - 1
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
        #  import ipdb; ipdb.set_trace()
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
