import os
import sys
import torch as tr
import torch.utils.data as dt
from os import path
from sklearn.metrics import precision_recall_curve, auc
from src.dataset import MaskedRNAGenerator
from src.model import RNADLM
from src.parameters import ParameterParser
from torch.utils.tensorboard import SummaryWriter


tr.backends.cudnn.deterministic = False
tr.backends.cudnn.benchmark = True

dev = tr.device("cuda:0")

sequence_len = 256
masked_proportion = 1/8
batch_size = 128

sw = SummaryWriter("runs/baseline")

def main(argv):
    pp = ParameterParser(argv)

    dataset = MaskedRNAGenerator(
        fasta_files = ["train_data/" + fn for fn in os.listdir('train_data/')],
        sequence_len = sequence_len,
        mask_lens=[1],
        masked_proportion=masked_proportion,
        )

    sampler = dt.RandomSampler(
        dataset,
        replacement=True,
        num_samples=batch_size
    )

    data_loader = dt.DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        pin_memory=True
        )

    model = RNADLM(dev, class_weights=dataset.class_weights)
    if path.exists("model.pmt"):
        model.load("model.pmt")
        print("Parameters from model.pmt loaded")
    model.train()

    epoch, last_save = 0, 0
    last_improvement = 0
    best_loss = float('inf')
    early_stop = float('inf')
    while last_improvement < early_stop:
        mean_loss, mean_acc, mean_bacc = 0, 0, 0
        for sequence, mask in data_loader:
            loss, acc, bacc = model.train_step(sequence, mask)
            loss /= len(data_loader)

            loss.backward()

            mean_loss += loss
            mean_acc  += acc / len(data_loader)
            mean_bacc += bacc / len(data_loader)
        model.optimizer_step()

        last_improvement += 1
        if mean_loss < best_loss and epoch - last_save > 128:
            last_save = epoch
            best_loss = mean_loss
            last_improvement = 0
            model.save("model.pmt")

        sw.add_scalar('Masking/Cross-entropy loss', mean_loss, epoch)
        sw.add_scalar('Masking/Accuracy', mean_acc, epoch)
        sw.add_scalar('Masking/Adjusted balanced accuracy', mean_bacc, epoch)
        epoch += 1
    sw.close()

if __name__ == "__main__":
    main(sys.argv[1:])
