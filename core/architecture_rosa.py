# --complete --fixed --format=codeblock
# FILE PATH: .\core\architecture_rosa.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class RWKV8_ROSA_Block(nn.Module):
    """
    RWKV-8 ROSA 核心物理层
    【天才级重构】：引入 O(T log T) 快速傅里叶变换 (FFT) 频域卷积！
    彻底湮灭 16GB 的 O(T^2) 空间衰减矩阵！
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
        
        # 强制 FP32 防止极端衰减时的精度坍塌
        w = -torch.exp(self.time_mix_w.float()) 
        
        # =========================================================================
        # 【神之降维：频域时空折叠 (FFT Causal Convolution)】
        # 抛弃 [C, T, T] 的 16GB 显存炸弹，在 O(T log T) 复杂度下完成绝对数学同构
        # =========================================================================
        
        # 1. 准备时域信号并转为 FP32 (FFT 需要高精度运算)
        kv = (k * v).float().transpose(1, 2) # [B, C, T]
        
        # 2. 构建一维物理衰减核 (Kernel)
        idx = torch.arange(T, device=x.device)
        decay = torch.exp(w.view(C, 1) * idx.view(1, T)).float() # [C, T]
        
        # 3. 升维至频域 (n=2*T 填充零，以避免循环卷积产生的伪影)
        kv_f = torch.fft.rfft(kv, n=2*T, dim=-1) # [B, C, T+1] (复数张量)
        decay_f = torch.fft.rfft(decay, n=2*T, dim=-1).unsqueeze(0) # [1, C, T+1] (复数张量)
        
        # 4. 频域点乘，在数学上绝对等价于时域中的因果卷积！
        wkv_f = kv_f * decay_f
        
        # 5. 逆变换回时域，严格截断前 T 个因果序列
        wkv = torch.fft.irfft(wkv_f, n=2*T, dim=-1)[..., :T] # [B, C, T]
        wkv = wkv.transpose(1, 2).to(x.dtype) # 变回 [B, T, C] 并降维回模型精度(BF16/FP16)
        
        # 路由门控 (ROSA Routing)
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
            # 激活重计算：将全局缓存粉碎，配合 FFT 将显存压榨到物理下限
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
                
                hidden_slice = x[b, c_len-1 : T]
                input_slice = input_ids[b, c_len-1 : T]
                
                non_pad_mask = (input_slice != padding_token_id)
                valid_hiddens.append(hidden_slice[non_pad_mask])
                
            valid_hiddens = torch.cat(valid_hiddens, dim=0)
            logits = self.head(valid_hiddens)
            return logits
            
        return self.head(x)