import fire
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from datasets import load_from_disk

from models.dense_model import get_model_config, DenseGPT
from lm_assemble.utils.loggers import CSVLogger
from datatools.pytorch_datasets import OnebyOne
from lm_assemble.utils.schedulers import get_cosine_annealing

def save_model(model: nn.Module, filename: str):
    torch.save(model.state_dict(), filename)

def load_pretrained_model_for_classification(pretrained_model_path, model, config, num_classes):
    pretrained_state_dict = torch.load(pretrained_model_path)

    # Optional: Remove proj_out layer weights from the pretrained state dict if necessary
    pretrained_state_dict = {key: value for key, value in pretrained_state_dict.items() if not key.startswith('proj_out')}

    model.load_state_dict(pretrained_state_dict, strict=False)
    model.proj_out = nn.Linear(config.dim, num_classes)

    return model

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
    logits = torch.mean(logits, dim=1)
    loss = F.cross_entropy(logits, y.view(-1))
    return loss

@torch.inference_mode()
def loss_and_eval_fn(model, x, y):
    target = y.view(-1)
    logits = model(x)
    logits = torch.mean(logits, dim=1)
    loss = F.cross_entropy(logits, target)
    classification = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
    accuracy = (classification == target).float().mean()
    return loss.item(), accuracy.item()

def eval_model(eval_steps, model, train_iter, test_iter):
    metrics = torch.empty(eval_steps, 4)
    for i in range(eval_steps):
        x_train, y_train = next(train_iter)
        x_test, y_test = next(test_iter)

        train_loss, train_acc = loss_and_eval_fn(model, x_train, y_train)
        test_loss, test_acc = loss_and_eval_fn(model, x_test, y_test)
    
        metrics[i] = torch.tensor([train_loss, train_acc, test_loss, test_acc])

    return metrics.mean(0)

def train(
    model_size: str = 'tiny',
    pretrained_model: str | None = 'saved_models/pretrained_tiny_long_cntx',
    save_final_model: str = 'saved_models/agGPT',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: str = 'float32',
    steps: int = 100,
    grad_accum_steps: int = 1,
    eval_every_n: int = 25,
    eval_steps: int = 100,
    max_lr: float = 1e-3,
    min_lr: float = 1e-4,
    warmup_steps: int = 10,
    resume: bool = False,
    checkpoint_dir: str = 'checkpoints/',
    checkpoint_file: str | None = 'checkpoint',
    save_every_n: int = 50,
    batch_size: int = 1

):

    assert batch_size == 1, "Batch Size must be one for one-by-one fine tuning"
    ds = load_from_disk('data/ag_news-ds')
    dtype = torch.float32 if dtype == 'float32' else None

    config = get_model_config(model_size)
    model = DenseGPT(config).to(device, dtype)

    if pretrained_model is not None:
        model = load_pretrained_model_for_classification(pretrained_model, model, config, 4)

    train_iter = iter(DataLoader(
        dataset=OnebyOne(ds['train'], max_cntx=config.cntx, device=device),
        batch_size=batch_size
    ))

    test_iter = iter(DataLoader(
        dataset=OnebyOne(ds['test'], max_cntx=config.cntx, device=device),
        batch_size=batch_size
    ))

    file_name = os.path.join(checkpoint_dir, checkpoint_file)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
    logger = CSVLogger('finetune_metrics.csv', fieldnames=['Step', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc'], resume=resume)

    scheduler = get_cosine_annealing(max_lr, min_lr, steps, warmup_steps)

    start_step = load_state(file_name, model, optimizer) if resume else 0

    # Training Loop
    for step in tqdm(range(start_step, steps), desc='Training'):
        
        scheduler(step, optimizer)

        if step % eval_every_n == 0 or step == steps - 1:
            avg_train_loss, avg_train_acc, avg_test_loss, avg_test_acc = eval_model(eval_steps, model, train_iter, test_iter)
            logger.log(
                Step=step,
                Train_Loss=avg_train_loss.item(),
                Train_Acc=avg_train_acc.item(),
                Test_Loss=avg_test_loss.item(),
                Test_Acc=avg_test_acc.item()
            )

        for _ in range(grad_accum_steps):
            x, y = next(train_iter)
            loss = loss_fn(model, x, y)
            loss = loss / grad_accum_steps
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % save_every_n == 0:
            save_state(model, optimizer, step, filename=file_name + f'_{step}')
    
    save_model(model, save_final_model)

if __name__== '__main__':
    fire.Fire(train)