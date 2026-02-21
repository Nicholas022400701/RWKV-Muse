import torch
from core.architecture_rosa import PianoMuseROSA

@torch.no_grad()
def generate_inspiration(model, context_tokens, generate_len=256, temp=0.85, top_p=0.90):
    # 1. 极致高效的前向预填充 (Prefilling)
    ctx_tensor = torch.tensor([context_tokens]).cuda()
    logits = model(ctx_tensor) 
    
    inspirations = []
    curr_token = logits[0, -1].argmax().item()
    
    # 2. 流式生成
    for _ in range(generate_len):
        inspirations.append(curr_token)
        
        curr_tensor = torch.tensor([[curr_token]]).cuda()
        out = model(curr_tensor)[0, -1]
        
        probs = torch.softmax(out / temp, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        remove_mask = cum_probs > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = 0
        
        probs[sorted_idx[remove_mask]] = 0.0
        probs = probs / probs.sum()
        curr_token = torch.multinomial(probs, 1).item()
            
    return inspirations

if __name__ == "__main__":
    print(f"[Generation] 启动纯血 RWKV-8 ROSA 原生推演器...")
    model = PianoMuseROSA(vocab_size=65536, n_layer=24, n_embd=1024).cuda().bfloat16()
    model.eval()
    
    mock_context = [12, 45, 88, 102, 33]
    result = generate_inspiration(model, mock_context)
    print(f"生成的灵感序列: {result[:20]}...")