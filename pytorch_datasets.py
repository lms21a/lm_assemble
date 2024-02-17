import torch
import torch.nn.functional as F
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

class LabeledDataset(IterableDataset):
    def __init__(self, ds, cntx, device='cpu'):
        super().__init__()

        self.data = ds
        self.cntx = cntx
        self.pad_id = 0
        self.device = device
        self.row_count = ds.num_rows  

    def __iter__(self):
        while True:
            idxs = list(range(self.row_count))
            shuffle(idxs)

            for ix in idxs:
                sample = self.data[ix]
                tokens = torch.tensor(sample['tokens'], dtype=torch.long)
                label = torch.tensor([sample['label']])

                # Pad tokens if they are shorter than the context length
                if sample['num_tokens'] < self.cntx:
                    tokens = F.pad(tokens, (0, self.cntx - tokens.size(0)), value=self.pad_id)
                
                # Truncate tokens if they are longer than the context length
                elif sample['num_tokens'] > self.cntx:
                    tokens = tokens[:self.cntx]

                assert len(tokens) == self.cntx, 'Tokens do not match expected shape. Investigate' # Can remove later

                yield tokens.to(device=self.device, non_blocking=True), label.to(device=self.device, non_blocking=True)