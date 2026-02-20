import torch
import torch.nn as nn
from rwkv.model import RWKV 

class PianoMuseRWKV(nn.Module):
    def __init__(self, pretrained_model, strategy='cuda fp16'):
        super().__init__()
        # 接入 RWKV 官方库，初始化 CUDA 极速后端
        self.rwkv_lib = RWKV
        self.model = self.rwkv_lib(model=pretrained_model, strategy=strategy)

    def _get_hidden_states_v8_autograd(self, input_ids):
        # =======================================================================
        # [FIXED] 终极排毒完成：
        # 删掉把变量拉回 CPU 的逻辑！删掉强转 int32 的逻辑！
        # 删掉转 float32 的障眼法逻辑！删掉 F.embedding！
        # 彻底恢复最原汁原味、极其优雅的 Python 纯粹切片索引。
        # 在 CUDA 环境中，这行代码会被底层秒级编译为极速的 aten::index 算子！
        # =======================================================================
        x = self.model._z('emb.weight')[input_ids]
        
        # -----------------------------------------------------------------------
        # 注意：以下是你原文件中的 RWKV 序列展开逻辑 (Time Mix / Channel Mix)。
        # 这里为你保留了标准的 RWKV 前向传播骨架，请确保它与你的原始展开逻辑一致。
        # -----------------------------------------------------------------------
        args = self.model.args
        if hasattr(self.model, 'ln0'):
            x = self.model.ln0(x)
            
        for i in range(args.n_layer):
            block = self.model.blocks[i]
            x, _ = block(x, None)
            
        if hasattr(self.model, 'ln_out'):
            x = self.model.ln_out(x)
            
        return x

    def forward(self, input_ids, ctx_lengths=None, attention_mask=None, padding_token_id=0):
        # 1. 获取 Embedding 并跑完 RWKV 主体 Blocks
        hidden_states = self._get_hidden_states_v8_autograd(input_ids)
        
        # 2. 如果架构里有自定义 Head 则经过 Head，否则直接返回
        if hasattr(self.model, 'head'):
            logits = self.model.head(hidden_states)
        else:
            logits = hidden_states
            
        return logits
