import sys
import torch as tr
import torch.utils.data as dt
from sklearn.metrics import precision_recall_curve, auc
from src.dataset import MaskedRNAGenerator
from src.model import RNADLM
from src.parameters import ParameterParser
from src.logger import log


tr.backends.cudnn.deterministic = False
tr.backends.cudnn.benchmark = True

def main(argv):
    pp = ParameterParser(argv)

    dataset = MaskedRNAGenerator(
        input_file="data/Caenorhabditis_elegans.WBcel235.dna_sm.toplevel.fa",
        sequence_len=32768,
        max_masked_len=1,
        masked_proportion=1/16
        )

    sampler = dt.RandomSampler(
        dataset,
        replacement=True,
        num_samples=100
    )

    data_loader = dt.DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
        pin_memory=True
        )

    model = RNADLM(tr.device("cpu"))
    log.write(
        "Number of parameters: %d" %
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    model.train()

    log.write('nbatch\tLoss\tAcc\tlast_imp\n')
    nbatch = 0
    train_loss = 100
    train_acc = 0
    best_train_loss = 100
    last_improvement = 0
    while last_improvement < 1000:
        for seq, idx, msk in data_loader:
            new_loss, new_acc = model.train_step(seq, idx, msk)
            train_loss = 0.99 * train_loss + 0.01 * new_loss
            train_acc =  0.99 * train_acc + 0.01 * new_acc

        model.lr_scheduler.step(train_loss)
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
