"""
RWKV Architecture Wrapper with Logit Physical Slicing & Pure PyTorch Autograd WKV.
[TLA+ Redesign] 
Restored native physical time-decay via WKV scan, keeping gradient flow perfectly intact.
Fixed LayerNorm Zero-Drift math bug for flawless boundary conditions.
Fixed Matmul broadcasting shape mismatch.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class PianoMuseRWKV(nn.Module):
    def __init__(self, model_path: str, strategy: str = 'cuda bf16'):
        super().__init__()
        
        try:
            from core.rwkv_training.rwkv_v8_model import RWKV_x070
            self.rwkv_lib = RWKV_x070
            self.using_training_model = True 
        except ImportError:
            raise ImportError("Critical Error: Missing RWKV_x070 training model.")
        
        import types
        model_args = types.SimpleNamespace()
        model_args.MODEL_NAME = model_path.replace('.pth', '') if model_path.endswith('.pth') else model_path
        model_args.n_layer = 12 
        model_args.n_embd = 768
        model_args.vocab_size = 65536
        model_args.head_size = 64
        
        self.model = self.rwkv_lib(model_args)
        
        # 动态解析自载入权重
        self.n_embd = self.model.n_embd
        self.n_layer = self.model.n_layer
        self.n_head = self.model.n_head
        self.head_size = getattr(self.model, 'head_size', 64)
        self.vocab_size = self.model.z['head_weight'].shape[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        ctx_lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        padding_token_id: int = 0
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        hidden_states = self._get_hidden_states_v8_autograd(input_ids)
        
        if return_hidden:
            return hidden_states
        
        if self.training and ctx_lengths is not None:
            valid_hiddens = []
            for b in range(batch_size):
                ctx_len = ctx_lengths[b].item()
                completion_hidden = hidden_states[b, ctx_len-1:, :]
                
                if attention_mask is not None:
                    completion_mask = attention_mask[b, ctx_len-1:]
                    non_pad_mask = completion_mask.bool()
                else:
                    completion_input_ids = input_ids[b, ctx_len-1:]
                    non_pad_mask = completion_input_ids != padding_token_id
                
                if non_pad_mask.any():
                    valid_hiddens.append(completion_hidden[non_pad_mask])
            
            if len(valid_hiddens) == 0:
                return torch.empty((0, self.vocab_size), device=input_ids.device, dtype=hidden_states.dtype)
            
            valid_hiddens = torch.cat(valid_hiddens, dim=0)
            return torch.matmul(valid_hiddens, self.model.z['head_weight'].T)
        
        hidden_flat = hidden_states.view(-1, self.n_embd)
        logits = torch.matmul(hidden_flat, self.model.z['head_weight'].T)
        return logits.view(batch_size, seq_len, self.vocab_size)
    
    def _get_hidden_states_v8_autograd(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.model._z('emb.weight')[input_ids]
        v_first = torch.empty_like(x)
        
        for i in range(self.n_layer):
            bbb = f'blocks_{i}_'
            
            # 【TLA+ 重构】：消灭 LayerNorm 零点漂移。
            # 必须先过 LN，然后再进行 Token Shift 平移并补 0，这样才能完美等价于 RWKV 的 state=0 边界条件！
            xx = F.layer_norm(x, (self.n_embd,), weight=self.model.z[bbb+'ln1_weight'], bias=self.model.z[bbb+'ln1_bias'])
            xx_prev_tmix = torch.cat([torch.zeros(batch_size, 1, self.n_embd, device=device, dtype=xx.dtype), xx[:, :-1, :]], dim=1)
            
            xx_out, v_first = self._batched_time_mix(xx, xx_prev_tmix, i, v_first)
            x = x + xx_out
            
            xx = F.layer_norm(x, (self.n_embd,), weight=self.model.z[bbb+'ln2_weight'], bias=self.model.z[bbb+'ln2_bias'])
            xx_prev_cmix = torch.cat([torch.zeros(batch_size, 1, self.n_embd, device=device, dtype=xx.dtype), xx[:, :-1, :]], dim=1)
            
            xx_out = self._batched_channel_mix(xx, xx_prev_cmix, i, input_ids)
            x = x + xx_out
            
        x = F.layer_norm(x, (self.n_embd,), weight=self.model.z['ln_out_weight'], bias=self.model.z['ln_out_bias'])
        return x
    
    def _batched_time_mix(self, x: torch.Tensor, x_prev: torch.Tensor, layer_id: int, v_first: torch.Tensor):
        """
        纯 PyTorch 推导的 WKV 时间衰减累加图模型。完整的 Autograd 梯度树反馈。
        修复了张量维度的广播断层。
        """
        B, T, C = x.shape
        H = self.n_head
        N = self.head_size
        att = f'blocks_{layer_id}_att_'
        
        dx = x_prev - x
        xr = x + dx * self.model.z[att+'x_r']
        xw = x + dx * self.model.z[att+'x_w']
        xk = x + dx * self.model.z[att+'x_k']
        xv = x + dx * self.model.z[att+'x_v']
        xa = x + dx * self.model.z[att+'x_a']
        xg = x + dx * self.model.z[att+'x_g']

        r = xr @ self.model.z[att+'receptance_weight']
        w = torch.tanh(xw @ self.model.z[att+'w1']) @ self.model.z[att+'w2']
        k = xk @ self.model.z[att+'key_weight']
        v = xv @ self.model.z[att+'value_weight']
        a = torch.sigmoid(self.model.z[att+'a0'] + (xa @ self.model.z[att+'a1']) @ self.model.z[att+'a2'])
        g = torch.sigmoid(xg @ self.model.z[att+'g1']) @ self.model.z[att+'g2']

        kk = torch.nn.functional.normalize((k * self.model.z[att+'k_k']).view(B, T, H, N), dim=-1, p=2.0).view(B, T, H*N)
        k = k * (1 + (a-1) * self.model.z[att+'k_a'])
        
        if layer_id == 0: 
            v_first = v
        else: 
            v = v + (v_first - v) * torch.sigmoid(self.model.z[att+'v0'] + (xv @ self.model.z[att+'v1']) @ self.model.z[att+'v2'])

        w = -torch.nn.functional.softplus(-(self.model.z[att+'w0'] + w)) - 0.5
        w_decay = torch.exp(w.float())

        state = torch.zeros(B, H, N, N, device=x.device, dtype=torch.float32)
        
        # 【致死点规避】：绝对的静态张量预分配。消灭 list.append 带来的 Graph Break。
        # 这是让华为 CANN 图引擎能够将 O(T) 循环完全展开并融合为巨型 BMM 算子的物理前提！
        out_static = torch.empty(B, T, H*N, device=x.device, dtype=x.dtype)
        
        v_ = v.view(B, T, H, N, 1).float()
        k_ = k.view(B, T, H, 1, N).float()
        kk_ = kk.view(B, T, H, N, 1).float()
        kka_ = (kk * a).view(B, T, H, 1, N).float()
        w_ = w_decay.view(B, T, H, 1, N)
        r_ = r.view(B, T, H, N, 1).float()

        # O(T) 在 NPU 静态图模式下会被 GE (Graph Engine) 自动解卷并扁平化
        for t in range(T):
            vk = v_[:, t] @ k_[:, t]
            ab = (-kk_[:, t]) @ kka_[:, t]
            state = state * w_[:, t] + (state @ ab) + vk
            out_static[:, t, :] = (state.to(dtype=x.dtype) @ r_[:, t]).view(B, H*N)
        
        out = out_static # 直接指向静态预分配内存
        
        out = torch.nn.functional.group_norm(out.view(B*T, H*N), num_groups=H, weight=self.model.z[att+'ln_x_weight'], bias=self.model.z[att+'ln_x_bias']).view(B, T, H*N)
        out = out + ((r * k * self.model.z[att+'r_k']).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, H*N)
        
        out = (out * g) @ self.model.z[att+'output_weight']
        return out, v_first
    
    def _batched_channel_mix(self, x: torch.Tensor, x_prev: torch.Tensor, layer_id: int, token_ids: torch.Tensor) -> torch.Tensor:
        ffn = f'blocks_{layer_id}_ffn_'
        dx = x_prev - x
        k = x + dx * self.model.z[ffn+'x_k']
        k = torch.relu(k @ self.model.z[ffn+'key_weight']) ** 2
        v = k @ self.model.z[ffn+'value_weight']
        
        enn_key = ffn+'enn_weight'
        if enn_key in self.model.z:
            enn = self.model.z[enn_key][token_ids]
            v = v * enn
        return v
    
    def generate(self, context_tokens: list, max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 0.9, top_k: int = 0) -> list:
        self.eval()
        with torch.no_grad():
            state = None
            
            # 【TLA+ 重构】：消灭智障的 Python O(T) 循环，启动底层原生并行 Prefill
            if len(context_tokens) > 1:
                if hasattr(self.model, 'forward_seq'):
                    _, state = self.model.forward_seq(context_tokens[:-1], state)
                else:
                    for token in context_tokens[:-1]:
                        _, state = self.model.forward([token], state)
            
            out, state = self.model.forward([context_tokens[-1]], state)
            
            generated = []
            current_token = self._sample_token(out, temperature, top_p, top_k)
            for _ in range(max_new_tokens):
                generated.append(current_token)
                out, state = self.model.forward([current_token], state)
                current_token = self._sample_token(out, temperature, top_p, top_k)
            return generated

    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9, top_k: int = 0) -> int:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, top_k_indices, top_k_probs)
            probs = probs / probs.sum()
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            probs[sorted_indices[sorted_indices_to_remove]] = 0.0
            probs = probs / probs.sum()
        return torch.multinomial(probs, 1).item()

def estimate_model_memory(n_layer, n_embd, vocab_size, batch_size, seq_len, precision='bf16'):
    bytes_per_param = {'fp32': 4, 'fp16': 2, 'bf16': 2}[precision]
    params = (n_layer * 4 * n_embd * n_embd) + (vocab_size * n_embd)
    param_memory = params * bytes_per_param / (1024**3)
    activation_memory = (batch_size * seq_len * n_embd * n_layer * 2 * bytes_per_param) / (1024**3)
    optimizer_memory = params * 2 * 4 / (1024**3) 
    gradient_memory = params * bytes_per_param / (1024**3)
    total = param_memory + activation_memory + optimizer_memory + gradient_memory
    return {
        'parameters_gb': round(param_memory, 2),
        'activations_gb': round(activation_memory, 2),
        'optimizer_gb': round(optimizer_memory, 2),
        'gradients_gb': round(gradient_memory, 2),
        'total_gb': round(total, 2),
        'model_params': params
    }
