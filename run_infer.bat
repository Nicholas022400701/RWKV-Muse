@echo off
echo ========================================================
echo [Genius Protocol] RWKV Piano Muse - O(1) Inference Engine
echo Target Env: C:\Users\nicho\anaconda3\python.exe
echo ========================================================

:: Quietly ensure 'uv' is installed in your exact conda environment
C:\Users\nicho\anaconda3\python.exe -m pip install uv -q

:: Use absolute path python module invocation to bypass PATH hell entirely
C:\Users\nicho\anaconda3\python.exe -m uv run --python C:\Users\nicho\anaconda3\python.exe infer_copilot.py ^
    --model_path ./models/best_model.pth ^
    --context_midi ./examples/context.mid ^
    --output_dir ./outputs ^
    --max_new_tokens 512 ^
    --temperature 0.85 ^
    --top_p 0.90

pause