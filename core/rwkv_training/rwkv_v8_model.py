import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time
from typing import List
import os

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F

MyModule = nn.Module
def MyFunction(x): return x
def MyStatic(x): return x

DTYPE = torch.bfloat16
HEAD_SIZE = 64

from torch.utils.cpp_extension import load
cuda_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
try:
    load(name="wkv7s", sources=[f"{cuda_dir}/wkv7s_op.cpp", f"{cuda_dir}/wkv7s.cu"], is_python_module=False,
         verbose=False, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
except Exception as e:
    pass

class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            if hasattr(torch.ops, 'wkv7s'):
                torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("CUDA backward not implemented. Use pure PyTorch WKV scan for training.")

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval()
        
        loaded_z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu', weights_only=True)
        
        # [Genius Re-design]: Dynamic Layer Deduction
        layer_keys = [int(k.split('.')[1]) for k in loaded_z.keys() if k.startswith('blocks.')]
        self.n_layer = max(layer_keys) + 1 if layer_keys else args.n_layer
        self.args.n_layer = self.n_layer

        self.n_head, self.head_size = loaded_z['blocks.0.att.r_k'].shape
        self.n_embd = loaded_z['emb.weight'].shape[1]
        
        # [Genius Re-design]: Break the ghost dict, wake up Autograd Graph!
        self.z = nn.ParameterDict()

        for k, v in loaded_z.items():
            if 'weight' in k and len(v.shape) == 2 and 'emb' not in k and 'head' not in k:
                if any(x in k for x in ['key.weight', 'value.weight', 'receptance.weight', 'output.weight']):
                    v = v.t()
            
            v = v.squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): v = v.flatten()

            safe_k = k.replace('.', '_')
            
            if k == 'emb.weight' and 'blocks.0.ln0.weight' in loaded_z:
                ln0_w = loaded_z['blocks.0.ln0.weight'].squeeze().to(dtype=DTYPE)
                ln0_b = loaded_z['blocks.0.ln0.bias'].squeeze().to(dtype=DTYPE)
                v = F.layer_norm(v, (self.n_embd,), weight=ln0_w, bias=ln0_b)
            if k.startswith('blocks.0.ln0.'): continue
                
            self.z[safe_k] = nn.Parameter(v, requires_grad=True)

        if 'blocks_0_att_a0' in self.z and 'blocks_0_att_v0' not in self.z:
            self.z['blocks_0_att_v0'] = nn.Parameter(self.z['blocks_0_att_a0'].detach().clone())
            self.z['blocks_0_att_v1'] = nn.Parameter(self.z['blocks_0_att_a1'].detach().clone())
            self.z['blocks_0_att_v2'] = nn.Parameter(self.z['blocks_0_att_a2'].detach().clone())

    def _z(self, name):
        return self.z[name.replace('.', '_')]

    def forward(self, idx, state, full_output=False):
        if state is None:
            state = [None for _ in range(self.n_layer * 3)]
            for i in range(self.n_layer):
                state[i*3+0] = torch.zeros(self.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+1] = torch.zeros((self.n_embd // self.head_size, self.head_size, self.head_size), dtype=torch.float, requires_grad=False, device="cuda")
                state[i*3+2] = torch.zeros(self.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")

        if isinstance(idx, list) or (isinstance(idx, torch.Tensor) and idx.dim() == 1 and idx.size(0) > 1):
            return self.forward_seq(idx, state, full_output)
        return self.forward_one(idx[0] if isinstance(idx, list) else idx, state)

    @MyFunction
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            x = self._z('emb.weight')[idx]
            v_first = torch.empty_like(x)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=self._z(bbb+'ln1.weight'), bias=self._z(bbb+'ln1.bias'))
                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(
                    i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    self._z(att+'x_r'), self._z(att+'x_w'), self._z(att+'x_k'), self._z(att+'x_v'), self._z(att+'x_a'), self._z(att+'x_g'),
                    self._z(att+'w0'), self._z(att+'w1'), self._z(att+'w2'), self._z(att+'a0'), self._z(att+'a1'), self._z(att+'a2'), self._z(att+'v0'), self._z(att+'v1'), self._z(att+'v2'),
                    self._z(att+'g1'), self._z(att+'g2'), self._z(att+'k_k'), self._z(att+'k_a'), self._z(att+'r_k'),
                    self._z(att+'receptance.weight'), self._z(att+'key.weight'), self._z(att+'value.weight'), self._z(att+'output.weight'),
                    self._z(att+'ln_x.weight'), self._z(att+'ln_x.bias'))
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=self._z(bbb+'ln2.weight'), bias=self._z(bbb+'ln2.bias'))
                enn_w = self._z(ffn+'enn.weight')[idx] if (ffn+'enn_weight').replace('.','_') in self.z else None
                xx, state[i*3+2] = RWKV_x080_CMix_one(xx, state[i*3+2], self._z(ffn+'x_k'), self._z(ffn+'key.weight'), self._z(ffn+'value.weight'), enn_w)
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=self._z('ln_out.weight'), bias=self._z('ln_out.bias'))
            return x @ self._z('head.weight'), state
        
    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            x = self._z('emb.weight')[idx]
            v_first = torch.empty_like(x)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=self._z(bbb+'ln1.weight'), bias=self._z(bbb+'ln1.bias'))
                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(
                    i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    self._z(att+'x_r'), self._z(att+'x_w'), self._z(att+'x_k'), self._z(att+'x_v'), self._z(att+'x_a'), self._z(att+'x_g'),
                    self._z(att+'w0'), self._z(att+'w1'), self._z(att+'w2'), self._z(att+'a0'), self._z(att+'a1'), self._z(att+'a2'), self._z(att+'v0'), self._z(att+'v1'), self._z(att+'v2'),
                    self._z(att+'g1'), self._z(att+'g2'), self._z(att+'k_k'), self._z(att+'k_a'), self._z(att+'r_k'),
                    self._z(att+'receptance.weight'), self._z(att+'key.weight'), self._z(att+'value.weight'), self._z(att+'output.weight'),
                    self._z(att+'ln_x.weight'), self._z(att+'ln_x.bias'))
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=self._z(bbb+'ln2.weight'), bias=self._z(bbb+'ln2.bias'))
                enn_w = self._z(ffn+'enn.weight')[idx] if (ffn+'enn_weight').replace('.','_') in self.z else None
                xx, state[i*3+2] = RWKV_x080_CMix_seq(xx, state[i*3+2], self._z(ffn+'x_k'), self._z(ffn+'key.weight'), self._z(ffn+'value.weight'), enn_w)
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=self._z('ln_out.weight'), bias=self._z('ln_out.bias'))
            return x @ self._z('head.weight'), state

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) 

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = -F.softplus(-(w0 + w)) - 0.5
    if hasattr(torch.ops, 'wkv7s'):
        xx = WKV_7.apply(state, r, w, k, v, -kk, kk*a)
    else:
        w_decay = torch.exp(w.float())
        state = state.clone()
        xx = torch.zeros(T, H*N, device=x.device, dtype=x.dtype)
        for t in range(T):
            r_, w_, k_, v_, kk_, a_ = r[t], w_decay[t], k[t], v[t], kk[t], a[t]
            vk = v_.view(H,N,1) @ k_.view(H,1,N)
            ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
            state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
            xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

    xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, x[-1,:], state, v_first

@MyStatic
def RWKV_x080_CMix_one(x, x_prev, x_k, K_, V_, E_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    res = (k @ V_)
    if E_ is not None: res = res * E_
    return res, x

@MyStatic
def RWKV_x080_CMix_seq(x, x_prev, x_k, K_, V_, E_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    res = (k @ V_)
    if E_ is not None: res = res * E_
    return res, x[-1,:]
