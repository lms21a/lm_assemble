import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import fire
from tqdm import tqdm

from src.utils.train_utils import save_state, loss_fn, eval_fn
from src.utils.schedulers import get_cosine_annealing
from src.models.hps import Hyperparameters
from src.models.llama_kinda import LlamaKinda, get_llama_config
from src.models.mod_transformer import MoDTransformer, get_mod_config
from datatools.data_processor import DataProcessor


def train(model: nn.Module, config: dict, data_processor: DataProcessor):
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    train_loader, val_loader = data_processor.prepare_dataloaders(generator)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=torch.cuda.is_available()
    )

    scheduler = get_cosine_annealing(config.max_lr, config.min_lr, config.steps, config.warmup_steps)

    for step in tqdm(range(config.steps), desc='Training'):
        
        if step % config.eval_interval == 0 or step == config.steps - 1:
            train_loss, val_loss = eval_fn(config.eval_steps, model, train_loader, val_loader)
            wandb.log({'Step': step, 'train_loss': train_loss.item(), 'val_loss': val_loss.item()})
        
        for _ in range(config.grad_accum_steps):

            try:
                x, y = next(train_iter)
            except:
                train_iter = iter(train_loader)
                x,y = next(train_iter)
            
            loss = loss_fn(model, x, y)
            loss = loss / config.grad_accum_steps
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler(step, optimizer)

        if step % config.save_interval == 0:
            save_state(model, optimizer, step, f'checkpoints/checkpoint_{step}')

def run(
        model: str = 'reg',
        dataset: str = 'lex',
        batch_size: int = 32,
        steps: int = 1000,
        eval_interval: int = 100,
        eval_steps: int = 100,
        save_interval: int = 500,
        grad_accum_steps: int = 1,
        warmup_steps: int = 100,
        max_lr: float = 1e-3,
        min_lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.01,
        seed: int = 1,
        model_size: str = 'tiny',
    ):

    data_dict = {
        'lex': 'data/lex.tokens',
        'shakespeare': 'data/shakespeare_tokens.npy'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = Hyperparameters(
        steps=steps,
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        save_interval=save_interval,
        grad_accum_steps=grad_accum_steps,
        max_lr=max_lr,
        min_lr=min_lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        seed=seed,
        model_size=model_size,
        batch_size=batch_size,
        dataset=dataset
    )


    model_config = get_llama_config(model_size)
    model = LlamaKinda(model_config)

    data_processor = DataProcessor(data_dict[dataset], model_config.max_cntx, batch_size, device)
    
    if device == 'cuda':
        model = torch.compile_model(model)

    model.to(device)
    model.print_model_size()

    wandb.init(project = 'LM_Assemble', config=config)

    train(model, config, data_processor)
    wandb.finish()

if __name__ == '__main__':
    fire.Fire(run)