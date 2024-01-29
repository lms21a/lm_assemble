import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from pytorch_datasets import AutoRegDataset
from torch.cuda.amp import GradScaler, autocast

import os
import argparse
from functools import partial
from tqdm import tqdm

from helper_func import (
    update_lr_linear_annealing, timing, load_checkpoint, 
    save_checkpoint, eval_model, loss_fn,
    count_parameters, timestamp
)

from models.hash_moe_model import MoeHashGPTConfig, MoeHashGPT
import wandb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training script for MoE Hash GPT Model")
    parser.add_argument('--steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--eval_every_n', type=int, default=10, help='Frequency of evaluations')
    parser.add_argument('--eval_steps', type=int, default=20, help='Number of steps for evaluation')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--check_point_interval', type=int, default=50, help='Checkpoint interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from the last checkpoint')
    parser.add_argument('--filename', type=str, help='Checkpoint filename to resume from')

    parser.add_argument('--vocab_size', type=int, default=1024, help='Vocabulary size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--cntx', type=int, default=32, help='Context size')
    parser.add_argument('--dim', type=int, default=32, help='Dimension size')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts')

    args = parser.parse_args()

    torch.manual_seed(1)

    train_arr = np.memmap('data/train_memmap.tokens', mode ='r', dtype = np.int16)
    test_arr = np.memmap('data/test_memmap.tokens', mode ='r', dtype = np.int16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = MoeHashGPTConfig(
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        cntx=args.cntx,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_experts=args.num_experts
    )

    model = MoeHashGPT(config=model_config).to(device=device)
    print(count_parameters(model))
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr)
    scheduler = partial(update_lr_linear_annealing, total_steps=args.steps, max_lr=args.max_lr, min_lr=args.max_lr/10)

    resume_from = load_checkpoint(model, optimizer, scaler, os.path.join(args.checkpoint_dir, args.filename)) if args.resume else 0 

    train_iter = iter(DataLoader(
    dataset=AutoRegDataset(train_arr, model_config.cntx, device),
    batch_size=model_config.batch_size 
    ))

    test_iter = iter(DataLoader(
        dataset=AutoRegDataset(test_arr, model_config.cntx, device),
        batch_size=model_config.batch_size
    ))

    model._init_params()
    wandb.init(project='LM_Assemble', name=f'moe_hash_v2_{timestamp()}')

    with timing('Training Time'):    
        for step in tqdm(range(resume_from, args.steps), desc = 'Training The Model'):
            
            if step % args.eval_every_n == 0 or (step == args.steps-1):
                eval_model(args.eval_steps, model, train_iter, test_iter)

            with autocast():
                for i in range(args.grad_accum_steps):
                    x, y = next(train_iter)
                    loss = loss_fn(model, x, y)
                    loss = loss / args.grad_accum_steps
                
                    scaler.scale(loss).backward()
        
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none = True)

            scheduler(optimizer, step)

            if step % args.check_point_interval == 0 or (step == args.steps-1):
                checkpoint = {
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict()
                }
                file_name = f'checkpoint_{step}'
                save_checkpoint(checkpoint, file_name)