# --complete --fixed --format=codeblock
# FILE PATH: .\core\architecture_rosa.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class RWKV8_ROSA_Block(nn.Module):
    """
    RWKV-8 ROSA (Routing Of Sequential Attention) 核心物理层
    纯原生实现，没有任何废库的黑盒封装。
    """
    def __init__(self, n_embd, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.time_mix_r = nn.Linear(n_embd, n_embd, bias=False)
        self.time_mix_k = nn.Linear(n_embd, n_embd, bias=False)
        self.time_mix_v = nn.Linear(n_embd, n_embd, bias=False)
        self.time_mix_w = nn.Parameter(torch.ones(n_embd)) 
        
        self.rosa_router = nn.Linear(n_embd, n_embd, bias=False) 
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.channel_mix_k = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.channel_mix_v = nn.Linear(n_embd * 4, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        
        # --- Time Mixing ---
        xx = self.ln1(x)
        r = self.time_mix_r(xx)
        k = self.time_mix_k(xx)
        v = self.time_mix_v(xx)
        
        w = -torch.exp(self.time_mix_w.float()) 
        
        # 并行衰减矩阵
        idx = torch.arange(T, device=x.device)
        decay_matrix = torch.exp(w.view(C, 1, 1) * (idx.view(1, T, 1) - idx.view(1, 1, T)).clamp(min=0))
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T)
        decay_matrix = decay_matrix * causal_mask

        kv = k * v
        wkv = torch.einsum('ctj, bcj -> btc', decay_matrix, kv.transpose(1, 2))
        
        # 路由门控
        routed_wkv = wkv * torch.sigmoid(self.rosa_router(xx))
        x = x + r * routed_wkv
        
        # --- Channel Mixing ---
        xx = self.ln2(x)
        k_cm = torch.relu(self.channel_mix_k(xx)) ** 2
        x = x + self.channel_mix_v(k_cm)
        
        return x

class PianoMuseROSA(nn.Module):
    def __init__(self, vocab_size=65536, n_layer=24, n_embd=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([RWKV8_ROSA_Block(n_embd, i) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
    def forward(self, input_ids, ctx_lengths=None, padding_token_id=0):
        x = self.emb(input_ids)
        
        for block in self.blocks:
            # 极限降维：阻断 Autograd 的 T^2 显存全局缓存
            x = checkpoint(block, x, use_reentrant=False)
            
        x = self.ln_out(x)
        
        if self.training and ctx_lengths is not None:
            # 【TLA+ 重设计：带 Padding 物理剔除的绝对安全切片】
            B, T, D = x.size()
            valid_hiddens = []
            
            for b in range(B):
                c_len = ctx_lengths[b]
                if isinstance(c_len, torch.Tensor):
                    c_len = c_len.item()
                
                # 1. 切出 Completion 区域的 Hidden States 和对应的 Input IDs
                hidden_slice = x[b, c_len-1 : T]
                input_slice = input_ids[b, c_len-1 : T]
                
                # 2. 严格提取非 Pad 掩码 (与 train_parallel 绝对同源)
                non_pad_mask = (input_slice != padding_token_id)
                
                # 3. 物理湮灭 Padding 的隐藏状态，最大化压榨 LM Head 显存
                valid_hiddens.append(hidden_slice[non_pad_mask])
                
            # [Valid_Tokens_In_Batch, D]
            valid_hiddens = torch.cat(valid_hiddens, dim=0)
            logits = self.head(valid_hiddens)
            return logits
            
        return self.head(x)