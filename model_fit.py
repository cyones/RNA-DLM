import sys
import torch as tr
import torch.utils.data as dt
from os import path
from sklearn.metrics import precision_recall_curve, auc
from src.dataset import MaskedRNAGenerator
from src.model import RNADLM
from src.parameters import ParameterParser
from src.logger import log


tr.backends.cudnn.deterministic = False
tr.backends.cudnn.benchmark = True

dev = tr.device("cuda:0")

def main(argv):
    pp = ParameterParser(argv)

    dataset = MaskedRNAGenerator(
        input_file="data/Caenorhabditis_elegans.WBcel235.dna_sm.toplevel.fa",
        sequence_len=32768,
        max_masked_len=1,
        masked_proportion=1/8
        )

    sampler = dt.RandomSampler(
        dataset,
        replacement=True,
        num_samples=1024
    )

    data_loader = dt.DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        pin_memory=True
        )

    model = RNADLM(dev)
    if path.exists("model.pmt"):
        log.write("Loading model ./model.pmt\n")
        model.load("model.pmt")

    log.write(
        "Number of parameters: %d\n" %
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    model.train()

    log.write('nbatch\tLoss\tAcc\tlast_imp\n')
    nbatch = 0
    train_loss = 100
    train_acc = 0
    best_train_loss = 100
    last_improvement = 0
    early_stop = float('inf')
    while last_improvement < early_stop:
        for seq, idx, msk in data_loader:
            seq, idx, msk = seq.to(dev), idx.to(dev), msk.to(dev)
            new_loss, new_acc = model.train_step(seq, idx, msk)
            train_loss = 0.99 * train_loss + 0.01 * new_loss
            train_acc =  0.99 * train_acc + 0.01 * new_acc
            model.lr_scheduler.step()
        last_improvement += 1
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            last_improvement = 0
            model.save("model.pmt")

        log.write('%d\t%.2f\t%.2f\t%d\n' %
                (nbatch, train_loss, train_acc, last_improvement))
        nbatch += 1

if __name__ == "__main__":
    main(sys.argv[1:])
