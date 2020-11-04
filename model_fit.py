import os
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
        fasta_files = ["train_data/" + fn for fn in os.listdir('train_data/')],
        max_chromosome_num = 1,
        sequence_len = 1024,
        mask_lens=[1],
        masked_proportion=1/8,
        )

    sampler = dt.RandomSampler(
        dataset,
        replacement=True,
        num_samples=16
    )

    data_loader = dt.DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        pin_memory=True
        )

    model = RNADLM(dev, class_weights=dataset.class_weights)
    if path.exists("model.pmt"):
        log.write("Loading model ./model.pmt\n")
        model.load("model.pmt")

    log.write(
        "Number of parameters: %d\n" %
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    model.train()

    log.write('epoch\tLoss\tAcc\tBAcc\tlast_imp\n')
    epoch = 0
    best_loss = float('inf')
    last_improvement = 0
    early_stop = float('inf')
    while last_improvement < early_stop:
        mean_loss, mean_acc, mean_bacc = 0, 0, 0
        for msk, seq in data_loader:
            loss, acc, bacc = model.train_step(msk.to(dev), seq.to(dev))

            loss /= len(data_loader)
            acc /= len(data_loader)
            bacc /= len(data_loader)

            loss.backward()

            mean_loss += loss
            mean_acc += acc
            mean_bacc += bacc
        model.optimizer_step()

        last_improvement += 1
        if mean_loss < best_loss:
            best_loss = mean_loss
            last_improvement = 0
            model.save("model.pmt")

        log.write('\r%d\t%.4f\t%.4f\t%.4f\t%d\n' %
                (epoch, mean_loss, mean_acc, mean_bacc, last_improvement))
        epoch += 1

if __name__ == "__main__":
    main(sys.argv[1:])
