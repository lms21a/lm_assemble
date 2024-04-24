import math
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
import numpy as np
from random import shuffle
from typing import List, Any, Generator, Iterator
from datasets import DatasetDict
from src.utils.helper_func import flatten_list

class AutoRegMapped(Dataset):
    def __init__(
            self,
            data_file: str,
            max_cntx: int,
            device: str,
            mask_threshold: float | None = None,
            randomly_mask: float | None = None
    ):
        super().__init__()
        self.data_file = data_file
        self.max_cntx = max_cntx
        self.device = device
        self.randomly_mask = randomly_mask
        self.mask_threshold = mask_threshold

        self._determine_arr_type()
        self.num_tokens = self._get_num_tokens()
    
    def _determine_arr_type(self):
        if self.data_file.endswith('.npy'):
            self.data = np.load(self.data_file)
            
        elif self.data_file.endswith('.tokens'):
            self.data = np.memmap(self.data_file, dtype=np.int16, mode='r')
    
    def _get_num_tokens(self):
        num_tokens = len(self.data)
        num_tokens = ((num_tokens - 1) // self.max_cntx) * self.max_cntx + 1
        return num_tokens
    
    def __len__(self):
        total_seqs = math.ceil((self.num_tokens - 1) / self.max_cntx)
        return total_seqs
    
    def _randomly_mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(x.shape) < self.randomly_mask
        return x.masked_fill(mask, 0)  # Replace masked elements with 0 (assuming 0 is the unknown token)
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_cntx
        chunk = min(self.max_cntx, self.num_tokens - start_idx - 1)  # Ensure there's a next token for autoreg
        tensor = torch.tensor(self.data[start_idx:start_idx + chunk + 1], dtype=torch.int64, device=self.device)

        # Decide whether to mask based on mask_threshold
        if self.mask_threshold is not None and torch.rand(1).item() < self.mask_threshold:
            input_tensor = self._randomly_mask(tensor[:-1])
        else:
            input_tensor = tensor[:-1]
        
        return input_tensor, tensor[1:].clone()

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

                block = torch.from_numpy(self.ds[start:end]).to(dtype=torch.int64, device=self.device, non_blocking = True)
                x = block[:-1]
                y = block[1:].clone()
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

def make_autoreg(tokens: List[int], cntx: int, **kwargs) -> Generator[torch.Tensor, torch.Tensor, Any]:
    chunks = (len(tokens) // cntx) - 1
    pos = list(range(chunks))
    shuffle(pos)

    for i in pos:
        start = i * cntx
        end = start + cntx + 1 # Increase size by 1 to slice autoregressively at the end
        block = torch.tensor(tokens[start:end], **kwargs)
        x = block[:-1]
        y = block[1:]
        yield x, y
            
class HF_AutoReg(IterableDataset):

    def __init__(
            self,
            dataset: DatasetDict,
            cntx: int,
            num_shards: int = 4,
            **kwargs
    )-> None:
        
        super().__init__()
        
        self.dataset = dataset
        self.cntx = cntx
        self.num_shards = num_shards
        self.kwargs = kwargs
    
    def __iter__(self) -> Iterator:
        while True:
            for i in range(self.num_shards):
                current_shard = self.dataset.shard(self.num_shards, index=i)
                tokens = flatten_list(current_shard['tokens'])
                yield from make_autoreg(tokens, self.cntx, **self.kwargs)

class OnebyOne(IterableDataset):

    def __init__(self, dataset: DatasetDict, max_cntx: int, device: str = 'cpu'):
        super().__init__()
        self.dataset = dataset
        self.max_cntx = max_cntx
        self.device = device
        self.num_rows = dataset.num_rows

    def __iter__(self) -> Iterator:
        description_order = list(range(self.num_rows))
        shuffle(description_order)

        while True:
            for idx in description_order:
                obs = self.dataset[idx]
                tokens = torch.tensor(obs['tokens'], device=self.device)
                labels = torch.tensor([obs['label']], device=self.device)

                if len(tokens) > self.max_cntx:
                    tokens = tokens[:self.max_cntx]
                
                yield tokens, labels