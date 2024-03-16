import fire
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import os

from datasets import load_from_disk

from schedulers import get_cosine_annealing
from models.dense_model import get_model_config, DenseGPT
from loggers import CSVLogger
from pytorch_datasets import HF_AutoReg

def save_state(model: nn.Module, optimizer: Optimizer, step: int, filename: str= "checkpoints/"):
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

def loss_fn(model, x, y):
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
    return loss

@torch.inference_mode()
def eval_fn(eval_steps, model, train_iter, test_iter):
    train_losses = []
    test_losses = []

    for step in range(eval_steps):
        x_train, y_train = next(train_iter)
        x_test, y_test = next(test_iter)
        
        train_loss = loss_fn(model, x_train, y_train)
        train_losses.append(train_loss.item())
        
        test_loss = loss_fn(model, x_test, y_test)
        test_losses.append(test_loss.item())

    average_train_loss = sum(train_losses) / len(train_losses)
    average_test_loss = sum(test_losses) / len(test_losses)

    return average_train_loss, average_test_loss

def train(
    model_size: str = 'tiny',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: str = 'float32',
    steps: int = 100,
    grad_accum_steps: int = 1,
    eval_every_n: int = 25,
    eval_steps: int = 100,
    max_lr: float = 1e-3,
    min_lr: float = 1e-4,
    warmup_steps: int = 10,
    checkpoint_dir: str = 'checkpoints/',
    resume: bool = False,
    checkpoint_file: str | None = 'checkpoint',
    save_every_n: int = 50,
    batch_size: int = 32
):

    ds = load_from_disk('data/ag_news-ds')
    dtype = torch.float32 if dtype == 'float32' else None

    config = get_model_config(model_size)
    model = DenseGPT(config).to(device, dtype)

    train_iter = iter(DataLoader(
        dataset=HF_AutoReg(ds['train'], config.cntx),
        batch_size=batch_size
    ))

    test_iter = iter(DataLoader(
        dataset=HF_AutoReg(ds['test'], config.cntx),
        batch_size=batch_size
    ))

    file_name = os.path.join(checkpoint_dir, checkpoint_file)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
    logger = CSVLogger('loss_log.csv', fieldnames=['Step', 'Train_Loss', 'Test_Loss'], resume=resume)

    scheduler = get_cosine_annealing(max_lr, min_lr, steps, warmup_steps)

    start_step = load_state(file_name, model, optimizer) if resume else 0

    # Training Loop
    for step in tqdm(range(start_step, steps), desc='Training'):
        
        scheduler(step, optimizer)

        if step % eval_every_n == 0 or step == steps - 1:
            avg_train, avg_test = eval_fn(eval_steps, model, train_iter, test_iter)
            logger.log(
                Step=step,
                Train_Loss=avg_train,
                Test_Loss=avg_test
            )

        for _ in range(grad_accum_steps):
            x, y = next(train_iter)
            loss = loss_fn(model, x, y)
            loss = loss / grad_accum_steps
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % save_every_n == 0 or step == steps - 1:
            save_state(model, optimizer, step, filename=file_name + f'_{step}')


if __name__ == '__main__':
    fire.Fire(train)
    
