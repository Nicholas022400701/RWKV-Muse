@echo off
echo ========================================================
echo [Genius Protocol] RWKV Piano Muse - Parallel Training
echo Target Env: C:\Users\nicho\anaconda3\python.exe
echo ========================================================

:: Quietly ensure 'uv' is installed in your exact conda environment
C:\Users\nicho\anaconda3\python.exe -m pip install uv -q

:: Use absolute path python module invocation to bypass PATH hell entirely
C:\Users\nicho\anaconda3\python.exe -m uv run --python C:\Users\nicho\anaconda3\python.exe train_parallel.py ^
    --data_path ./data/processed/processed_dataset.jsonl ^
    --pretrained_model ./models/rwkv_base.pth ^
    --output_dir ./models ^
    --batch_size 4 ^
    --max_seq_len 1024 ^
    --epochs 10

pause