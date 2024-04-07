import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

def nearest_lower_divisor(number, divisor):
    if number % divisor == 0:
        return number
    else:
        return number - (number % divisor)

def save_state(model: nn.Module, optimizer: Optimizer, step: int, filename: str):
    state = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }

    torch.save(state, filename)

def load_state(filename: str, model: nn.Module, optimizer: Optimizer):

    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    step = checkpoint['step']
    
    return step

def loss_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
    return loss

def eval_fn(eval_steps: int, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> torch.Tensor:
    metrics = torch.empty((eval_steps, 2))
    reset_interval = min(len(train_loader), len(val_loader))

    model.eval()
    with torch.inference_mode():
        for step in range(eval_steps):
            if step % reset_interval == 0:
                train_iter = iter(train_loader)
                val_iter = iter(val_loader)
            
            x_train, y_train = next(train_iter)
            x_val, y_val = next(val_iter)
            
            train_loss = loss_fn(model, x_train, y_train)
            val_loss = loss_fn(model, x_val, y_val)
        
            metrics[step] = torch.tensor([train_loss.item(), val_loss.item()])
        
    model.train()
    return metrics.mean(0)