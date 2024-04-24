import torch
from torch.utils.data import DataLoader, random_split
from src.datatools.pytorch_datasets import AutoRegMapped

class DataProcessor:
    def __init__(
            self,
            data_file: str,
            batch_size: int,
            device: str,
            max_cntx: int,
            train_size: float = .8,
            shuffle: bool = True,
            mask_threshold: float | None = None,
            randomly_mask: float | None = None
    ):
        self.data_file = data_file
        self.max_cntx = max_cntx
        self.batch_size = batch_size
        self.train_size = train_size
        self.device = device
        self.shuffle = shuffle
        self.randomly_mask = randomly_mask
        self.mask_threshold = mask_threshold
    
    def prepare_dataloaders(self, generator: torch.Generator | None = None):
        data = AutoRegMapped(self.data_file, self.max_cntx, self.device, self.mask_threshold, self.randomly_mask)
        
        train_set, val_set = random_split(data, [self.train_size, 1 - self.train_size], generator=generator)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_loader, val_loader