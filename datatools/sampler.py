import torch
import torch.nn.functional as F

class Sampler:
    def __init__(self, tokenizer, temperature=1.0, k=5):
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.k = k
    
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        preds = logits[:, -1, :] / self.temperature

        topk_vals, topk_inds = torch.topk(preds, self.k, dim=-1)
        filtered_logits = torch.full_like(preds, float('-inf'))  
        filtered_logits.scatter_(dim=-1, index=topk_inds, src=topk_vals) 

        probs = F.softmax(filtered_logits, dim=-1)
        ids = torch.multinomial(probs, num_samples=1)
        return ids