import math
from torch.optim import Optimizer

def update_lr_linear_annealing(optimizer, current_step, max_lr, min_lr, total_steps):

    if current_step < total_steps:
        # Calculate the new learning rate
        new_lr = max_lr - (max_lr - min_lr) * (current_step / total_steps)

        # Update the optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def get_cosine_annealing(initial_lr: float = 0.01, min_lr: float = 0.001, total_steps: int = 10000, warm_up_steps: int = 1000):

    lr_list = []
    for current_step in range(total_steps):
        if current_step < warm_up_steps:
            
            lr = (initial_lr * current_step) / warm_up_steps
        else:
            
            adjusted_step = current_step - warm_up_steps
            adjusted_total_steps = total_steps - warm_up_steps
            cosine = math.cos(math.pi * adjusted_step / adjusted_total_steps)
            lr = min_lr + (initial_lr - min_lr) * (1 + cosine) / 2
        lr_list.append(lr)
    
    def cosine_annealing(current_step: int, optimizer: Optimizer):
        lr = lr_list[current_step] if current_step < total_steps else min_lr
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return cosine_annealing