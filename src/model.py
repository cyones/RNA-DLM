import torch as tr
import torch.nn as nn
from performer_pytorch import PerformerLM
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

        self.performer = PerformerLM(
                num_tokens = 4096,
                max_seq_len = 128,             # max sequence length
                dim = 256,                      # dimension
                depth = 6,                     # layers
                heads = 4,                      # heads
                causal = False,                 # auto-regressive or not
                nb_features = 64,              # number of random features, if not set, will default to (d * log(d))
                generalized_attention = False,  # defaults to softmax approximation, but can be set to True for generalized attention
                kernel_fn = nn.GELU(),          # the kernel function to be used, if generalized attention is on, defaults to Relu
                reversible = True,              # reversible layers, from Reformer paper
                ff_chunks = 10,                 # chunk feedforward layer, from Reformer paper
                use_scalenorm = False,          # use scale norm, from 'Transformers without Tears' paper
                use_rezero = False,             # use rezero, from 'Rezero is all you need' paper
                ff_glu = True,                  # use GLU variant for feedforward
                emb_dropout = 0.1,              # embedding dropout
                ff_dropout = 0.1,               # feedforward dropout
                attn_dropout = 0.1,             # post-attn dropout
                local_attn_heads = 4,           # 4 heads are local attention, 4 others are global performers
                local_window_size = 32         # window size of local attention
                )

        self.loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = tr.optim.SGD(
            self.parameters(),
            lr=1e-4,
            momentum=0.9,
        )
        self.lr_scheduler = tr.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            scale_mode="exp_range",
            gamma=1.00,
            base_lr=1e-5,
            max_lr=1e-1,
            step_size_up=32,
            cycle_momentum=True,
            base_momentum=0.8,
            max_momentum=0.9
        )
        self.optimizer.zero_grad()
        self.to(device = self.device)
        self.requires_grad_(True)

    def forward(self, seq, mask=None):
        if mask is not None:
            mask = mask.to(device = self.device)
            x = self.performer(seq, mask=mask)
        else:
            x = self.performer(seq)
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
