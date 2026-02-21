import torch
import torch.nn as nn
from torch.nn import functional as F

class RWKV8_ROSA_Block(nn.Module):
    """
    RWKV-8 ROSA (Routing Of Sequential Attention) 核心物理层
    纯原生实现，没有任何废库的黑盒封装。
    """
    def __init__(self, n_embd, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        
        # ROSA 动态路由机制的参数
        self.ln1 = nn.LayerNorm(n_embd)
        self.time_mix_r = nn.Linear(n_embd, n_embd, bias=False)
        self.time_mix_k = nn.Linear(n_embd, n_embd, bias=False)
        self.time_mix_v = nn.Linear(n_embd, n_embd, bias=False)
        self.time_mix_w = nn.Parameter(torch.ones(n_embd)) # 指数衰减物理映射
        
        self.rosa_router = nn.Linear(n_embd, n_embd, bias=False) # v8 特有路由门控
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.channel_mix_k = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.channel_mix_v = nn.Linear(n_embd * 4, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        
        # --- Time Mixing (ROSA Attention) ---
        xx = self.ln1(x)
        r = self.time_mix_r(xx)
        k = self.time_mix_k(xx)
        v = self.time_mix_v(xx)
        
        # RWKV-8 独有的 ROSA Routing (支持梯度流的原生并行扫描表达)
        w = -torch.exp(self.time_mix_w.float()) 
        
        # 物理切片：利用 causal mask 构建并行衰减矩阵
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
        
    def forward(self, input_ids, ctx_lengths=None):
        x = self.emb(input_ids)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_out(x)
        
        if self.training and ctx_lengths is not None:
            # 【TLA+ 重设计：极致物理切片降维】
            B, T, D = x.size()
            valid_hiddens = []
            
            for b in range(B):
                c_len = ctx_lengths[b]
                if isinstance(c_len, torch.Tensor):
                    c_len = c_len.item()
                valid_hiddens.append(x[b, c_len-1 : T])
                
            valid_hiddens = torch.cat(valid_hiddens, dim=0)
            logits = self.head(valid_hiddens)
            return logits
            
        return self.head(x)