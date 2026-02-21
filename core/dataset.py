# --complete --fixed --format=codeblock
# FILE PATH: /content/RWKV-Muse/core/dataset.py
import json
import ast
import torch
from torch.utils.data import Dataset

def load_dataset(jsonl_path):
    """
    【极速数据泵】：将 JSONL 中的音乐流形直接载入 Host 内存
    """
    data = []
    print(f"[*] 正在物理读取音乐序列: {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # 【量子级容错】如果由于某种原因写入了单引号 dict 字符串，利用 AST 强行反序列化
                try:
                    data.append(ast.literal_eval(line))
                except Exception as e:
                    print(f"[警告] 第 {line_num+1} 行发生绝对坍塌，已物理丢弃: {e}")
    print(f"[+] 成功萃取 {len(data)} 个高维音乐切片！")
    return data

class CopilotDataset(Dataset):
    def __init__(self, data_pairs, max_seq_len=2048, tokenizer=None):
        """
        data_pairs: 包含 'context' 和 'completion' 的字典列表
        max_seq_len: 物理截断长度，防止显存 OOM
        """
        self.data = data_pairs
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer # 预留
        
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ctx_tokens = item['context']
        comp_tokens = item['completion']
        
        full_seq = ctx_tokens + comp_tokens
        
        # 【物理截断】：如果序列太长，优先保留 Completion，Context 强制保留最后 1/4 作为因果锚点
        # +1 是为了给自回归的 target 预留出一位向未来的绝对偏移空间
        if len(full_seq) > self.max_seq_len + 1:
            keep_ctx = min(len(ctx_tokens), max(2, self.max_seq_len // 4))
            ctx_tokens = ctx_tokens[-keep_ctx:]
            comp_tokens = comp_tokens[:self.max_seq_len + 1 - len(ctx_tokens)]
            full_seq = ctx_tokens + comp_tokens
            
        ctx_len = len(ctx_tokens)
        
        return {
            # 严格自回归：Target 永远是 Input 面向未来的绝对投影 (左移一位)
            "input_ids": torch.tensor(full_seq[:-1], dtype=torch.long),
            "target_ids": torch.tensor(full_seq[1:], dtype=torch.long),
            "ctx_len": ctx_len
        }

def collate_fn(batch):
    """
    【二维张量对齐器】：将变长的音乐切片在 Batch 维度上用 0 (Padding) 完美对齐
    这正是 architecture_rosa 和 train_parallel 能够通过 mask 物理剔除 Padding 的前置条件！
    """
    # 探测当前 Batch 中最长的物理序列天花板
    max_len = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)
    
    # 建立全 0 矩阵 (0 就是我们的 Padding Token，它不代表任何有效音符)
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    target_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    ctx_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # 将真实数据填入左侧 (Left-aligned)
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        target_ids[i, :seq_len] = item["target_ids"]
        ctx_lengths[i] = item["ctx_len"]
        
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "ctx_lengths": ctx_lengths
    }