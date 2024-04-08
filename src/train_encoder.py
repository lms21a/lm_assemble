import torch
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_from_disk
from datatools.pytorch_datasets import LabeledDataset
from torch.utils.data import DataLoader

from models.encoder_transformer import MoeEncoderConfig, MoeEncoderTransformer

from tqdm import tqdm
from functools import partial
from utils.loggers import CSVLogger
from utils.helper_func import plot_csv

def loss_fn(model, x, y):
    logits = model(x)
    loss = F.cross_entropy(logits, y.view(-1), ignore_index=0)
    return loss

@torch.inference_mode()
def loss_and_eval_fn(model, x, y):
    target = y.view(-1)
    logits = model(x)
    loss = F.cross_entropy(logits, target, ignore_index=0)
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

if __name__ == '__main__':
    ds = load_from_disk('data/ag_news-ds')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = .001
    steps = 2000
    eval_every_n = steps // 10
    eval_steps = 100
    grad_accum_steps = 4

    config = MoeEncoderConfig(
        vocab_size=512,
        batch_size=32,
        cntx=128,
        dim=64,
        num_heads=4,
        num_layers=4,
        num_experts=8,
        num_classes=4
    )

    train_iter = iter(DataLoader(
        LabeledDataset(
            ds=ds['train'],
            cntx=config.cntx,
            device=device
        ),
        batch_size=config.batch_size
    ))

    test_iter = iter(DataLoader(
        LabeledDataset(
            ds=ds['test'],
            cntx=config.cntx,
            device=device
        ),
        batch_size=config.batch_size
    ))

    model = MoeEncoderTransformer(config=config).to(device=device)
    model.print_model_size()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    logger = CSVLogger('loss_log.csv', fieldnames=['Step', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc'])
    eval_model_fn = partial(eval_model, model=model, train_iter=train_iter, test_iter=test_iter)

    for step in tqdm(range(steps), desc = 'Training The Model'):
        if (step % eval_every_n == 0) or (step == steps-1):
            train_loss, train_acc, test_loss, test_acc = eval_model_fn(eval_steps)
            logger.log(
                Step=step,
                Train_Loss=train_loss.item(),
                Train_Acc=train_acc.item(),
                Test_Loss=test_loss.item(),
                Test_Acc=test_acc.item()
            )
        
        for mini_step in range(grad_accum_steps):
            x, y = next(train_iter)
            loss = loss_fn(model, x, y)
            loss = loss / grad_accum_steps
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    plot_csv('loss_log.csv')