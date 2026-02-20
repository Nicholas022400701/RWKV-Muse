@echo off
echo ========================================================
echo [Genius Protocol] RWKV Piano Muse - Tensor Tokenization
echo Target Env: C:\Users\nicho\anaconda3\python.exe
echo ========================================================

set PY=C:\Users\nicho\anaconda3\python.exe

:: Quietly ensure dependencies
%PY% -m pip install miditok symusic mido datasets -q

%PY% -m uv run --python %PY% scripts/preprocess_data.py ^
    --midi_dir ./data/raw_midi ^
    --output_dir ./data/processed ^
    --vocab_size 65536 ^
    --n_context_bars 4 ^
    --n_completion_bars 2 ^
    --step 2

pause