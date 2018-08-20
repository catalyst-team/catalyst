import numpy as np
import tqdm
import torch


class CorpusLoader:
    """
    Language model iterator that iterates through batches
        that are of length N(bptt, bptt_std)
    The first batch returned is always bptt+bptt_std^2; the max possible width.
    By this way pytorch allocates cuda memory in order
    to prevent multiple buffers from being created as the batch width grows.
    """
    def __init__(
            self, txt_data, encode_fn,
            bs, bptt_mean, bptt_std=5,
            verbose=False):
        txt_data = tqdm.tqdm(txt_data) if verbose else txt_data
        data = list(map(encode_fn, txt_data))
        data = [x for y in data for x in y]
        data = torch.LongTensor(data)
        data = CorpusLoader.batchify(data, bs)
        self.data = data
        self.bs = bs
        self.bptt_mean = bptt_mean
        self.bptt_std = bptt_std
        self.i, self.iter = 0, 0
        self.n = len(self.data)

    @staticmethod
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn"t cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt_mean + self.bptt_std * self.bptt_std
            else:
                bptt = (
                    self.bptt_mean
                    if np.random.random() < 0.95
                    else self.bptt_mean / 2.)
                seq_len = max(
                    self.bptt_std,
                    int(np.random.normal(bptt, self.bptt_std)))
            batch = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield batch

    def get_batch(self, idx, seq_len):
        data = self.data[idx:idx + seq_len]
        target = self.data[idx + 1:idx + 1 + seq_len]
        dct = {
            "txt": data,
            "trg": target
        }
        return dct

    def __len__(self):
        return self.n // self.bptt_mean - 1
