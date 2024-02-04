from models.hash_moe_model import MoeHashGPT, MoeHashGPTConfig
from tokenizer import Tokenizer
from sampler import Sampler

import torch

if __name__ == '__main__':
    USE_PREFIX = True
    max_gen_len = 20
    
    tokenizer = Tokenizer('saved_models/spm_model.model')
    tokenizer.load_tokenizer(add_bos=True)

    sampler = Sampler(tokenizer, temperature=.5, k=10)

    model = MoeHashGPT(MoeHashGPTConfig(
    vocab_size=1024,
    batch_size=32,
    cntx=32,
    dim=32,
    num_heads=4,
    num_experts=4,
    num_layers=4
    ), sampler)

    model.load_state_dict(torch.load('saved_models/test_checkpoint')['model_state'])

    if USE_PREFIX:
        prefix_text = input('Please Enter your prefix text:\n')
        output = model.generate(prefix_text=prefix_text, max_len=max_gen_len)

    else:
        output = model.generate(max_len=max_gen_len)

    print(output)

