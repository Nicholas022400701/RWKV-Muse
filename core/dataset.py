import torch
from torch.utils.data import Dataset

class CopilotDataset(Dataset):
    def __init__(self, tokenized_data, max_seq_len=2048):
        self.data = tokenized_data
        self.max_seq_len = max_seq_len
        
    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ctx_tokens = item['context']
        comp_tokens = item['completion']
        
        full_seq = ctx_tokens + comp_tokens
        
        if len(full_seq) > self.max_seq_len:
            keep_ctx = min(len(ctx_tokens), max(2, self.max_seq_len // 4))
            ctx_tokens = ctx_tokens[-keep_ctx:]
            comp_tokens = comp_tokens[:self.max_seq_len - len(ctx_tokens)]
            full_seq = ctx_tokens + comp_tokens
            
        ctx_len = len(ctx_tokens)
        
        return {
            "input_ids": torch.tensor(full_seq[:-1], dtype=torch.long),
            "target_ids": torch.tensor(full_seq[1:], dtype=torch.long),
            "ctx_len": ctx_len
        }