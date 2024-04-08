import torch
from torch.utils.data import DataLoader, random_split
from src.datatools.pytorch_datasets import AutoRegMapped

class DataProcessor:
    def __init__(
            self,
            data_file: str,
            max_cntx: int,
            batch_size: int,
            device: str,
            train_size: float = .8,
            shuffle: bool = True
    ):
        self.data_file = data_file
        self.max_cntx = max_cntx
        self.batch_size = batch_size
        self.train_size = train_size
        self.device = device
        self.shuffle = shuffle
    
    def prepare_dataloaders(self, generator: torch.Generator | None = None):
        data = AutoRegMapped(self.data_file, self.max_cntx, self.device)
        
        train_set, val_set = random_split(data, [self.train_size, 1 - self.train_size], generator=generator)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_loader, val_loader