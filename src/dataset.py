import torch as tr
import random as rn
from Bio import SeqIO
from src.logger import log
from torch.utils.data import Dataset

class MaskedRNAGenerator(Dataset):
    def __init__(
        self,
        input_file,
        sequence_len,
        max_masked_len,
        masked_proportion,
        max_chromosome_num=None
        ):
        self.sequence_len = sequence_len
        self.max_masked_len = max_masked_len
        self.masked_proportion = masked_proportion
        self.chromosome = []

        self.total_len = 0
        nseqs = 0
        with open(input_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq.lower().transcribe())
                if len(seq) < self.sequence_len:
                    log.write(f"Sequence {record.name} too short, skipped...\n")
                tns = tr.LongTensor([self.seq2num[n] for n in seq])
                self.chromosome.append(tns)
                self.total_len += len(seq)
                nseqs += 1
                if max_chromosome_num and nseqs >= max_chromosome_num:
                    break
        log.write(f"Loaded dataset with {self.total_len} bases\n")

    def __len__(self):
        return self.total_len - self.sequence_len

    def __getitem__(self, index):
        chromosome_idx = 0
        while index > len(self.chromosome[chromosome_idx]):
            index -= len(self.chromosome[chromosome_idx])
            chromosome_idx += 1
        seq = self.chromosome[chromosome_idx][index:(index+self.sequence_len)]
        seq = tr.nn.functional.pad(seq, [0, self.sequence_len-len(seq)])
        
        mask_len = rn.randint(1, self.max_masked_len)
        mask_number = int((self.masked_proportion * self.sequence_len) // mask_len)
        mask_idx = tr.randint(high=self.sequence_len-mask_len, size=[mask_number])

        mask_idx_starts = mask_idx.clone()
        for i in range(1, mask_len):
            mask_idx = tr.cat([mask_idx, mask_idx_starts+i-1])
        masked_sequence = seq[mask_idx]
        seq[mask_idx] = 0

        return seq, mask_idx, masked_sequence

    seq2num = {'a': 1, 'c': 2, 'g': 3, 'u': 4, 'w': 0,
               's': 0, 'k': 0, 'm': 0, 'y': 0, 'r': 0,
               'b': 0, 'd': 0, 'h': 0, 'v': 0, 'n': 0}
