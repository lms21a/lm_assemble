import torch
import torch.nn.functional as F 

def greedy_sample(model, sample):
    with torch.inference_mode():
        logits = model(sample)
        preds = logits[:, -1, :]
        ids = torch.multinomial(F.softmax(preds, dim=-1), num_samples=1)
        return ids

def top_k_sample(model, sample, k=10):
    with torch.inference_mode():
        logits = model(sample)  
        preds = logits[:, -1, :]

        topk_vals, topk_inds = torch.topk(preds, k, dim=-1)
        filtered_logits = torch.full_like(preds, float('-inf'))  
        filtered_logits.scatter_(dim=-1, index=topk_inds, src=topk_vals) 

        probs = F.softmax(filtered_logits, dim=-1)
        ids = torch.multinomial(probs, num_samples=1)
        return ids
    
def temperature_scaled_sampling(model, sample, temperature=1.0):
    with torch.inference_mode():
        logits = model(sample)  
        logits = logits[:, -1, :] / temperature  
        probabilities = F.softmax(logits, dim=-1)  
        sampled_token_id = torch.multinomial(probabilities, num_samples=1) 
        return sampled_token_id