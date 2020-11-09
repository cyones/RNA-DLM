import torch as tr
import random as rn
import sentencepiece as spm
from Bio import SeqIO
from src.logger import log
from torch.utils.data import Dataset
from tqdm import tqdm


MASKED_IDX = 4096

class MaskedRNAGenerator(Dataset):
    def __init__(
        self,
        fasta_files,
        sequence_len,
        mask_lens,
        masked_proportion,
        max_chromosome_num=None
        ):
        self.sequence_len = sequence_len
        self.mask_lens = mask_lens
        self.masked_proportion = masked_proportion
        self.chromosome = []

        sp = spm.SentencePieceProcessor(model_file='tokenizers/cel_bpe_4096.model')

        self.total_len = 0
        nseqs = 0
        for filename in fasta_files:
            print(f"Loading file {filename}")
            with open(filename, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    print(f"Tokenizing sequence {record.name}")
                    seq = str(record.seq.upper())
                    chunk_size = 1000000
                    tokens = []
                    for chunk in tqdm([seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]):
                        tokens.extend(sp.encode(chunk)[1:])
                    if len(tokens) < self.sequence_len:
                        log.write(f"Sequence {record.name} too short, skipped...\n")
                    tns = tr.LongTensor(tokens)
                    self.chromosome.append(tns)
                    self.total_len += (len(tokens) - self.sequence_len)
                    nseqs += 1
                    if max_chromosome_num and nseqs >= max_chromosome_num:
                        break
        self.class_weights = 1 / (tr.bincount(tr.cat([seq for seq in self.chromosome])) + 10)
        self.class_weights /= self.class_weights.sum()
        log.write(f"Loaded dataset with {self.total_len + self.sequence_len} tokens\n")

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        chromosome_idx = 0
        while index > (len(self.chromosome[chromosome_idx]) - self.sequence_len):
            index -= (len(self.chromosome[chromosome_idx]) - self.sequence_len)
            chromosome_idx += 1
        sequence = self.chromosome[chromosome_idx][index:(index+self.sequence_len)]

        mask_len = rn.choice(self.mask_lens)
        mask_number = int((self.masked_proportion * self.sequence_len) // mask_len)
        mask_idx = tr.randint(high=self.sequence_len-mask_len, size=[mask_number])

        mask_idx_starts = mask_idx.clone()
        mask_idx = [mask_idx]
        for i in range(1, mask_len):
            mask_idx.append(mask_idx_starts+i-1)
        mask_idx = tr.cat(mask_idx)
        # Do not mask unknown bases
        mask_idx = mask_idx[ sequence[mask_idx] > 2]

        masked_sequence = sequence.clone()
        masked_sequence[mask_idx] = MASKED_IDX

        return masked_sequence, sequence
