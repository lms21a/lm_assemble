# TODO Rename arrays
import torch
from torch.utils.data import IterableDataset
from random import shuffle

class AutoRegDataset(IterableDataset):
    def __init__(self, ds, cntx, device):
        super().__init__()
        
        self.ds = ds
        self.cntx = cntx

        self.chunks = (len(ds) // cntx) - 1 # Remove Partial Chunk
        self.device = device

    def __iter__(self):
        
        while True:
            idxs = list(range(self.chunks))
            shuffle(idxs)

            for i in idxs:
                start = i * self.cntx
                end = start + self.cntx + 1 # Increase size by 1 to slice autoregressively at the end

                block = torch.from_numpy(self.ds[start:end].copy()).to(dtype=torch.int64, device=self.device, non_blocking = True)
                x = block[:-1]
                y = block[1:]
                yield x, y