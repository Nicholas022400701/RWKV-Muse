"""
MIDI Tokenization using REMI (Revamped MIDI-derived events) representation.
Provides structured token conversion for piano music completion.
[Genius Protocol] Bulletproof Version-Agnostic Engine to survive MidiTok v3 API apocalypse.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
try:
    import miditok
    from miditok import REMI, TokenizerConfig
except ImportError:
    print("[WARNING] miditok not installed. Install with: pip install miditok")
    REMI = None


class PianoTokenizer:
    """
    MIDI tokenizer specifically designed for piano music completion tasks.
    Engineered with absolute backward/forward compatibility to bypass MidiTok API breaks.
    """
    
    def __init__(self, vocab_size: int = 65536, max_bar_embedding: int = 300):
        if REMI is None:
            raise ImportError("miditok is required. Install with: pip install miditok")
        
        config = TokenizerConfig(
            num_velocities=32,
            use_chords=False,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=False,
            beat_res={(0, 4): 8, (4, 12): 4},
            num_tempos=32,
            tempo_range=(40, 250),
        )
        
        self.tokenizer = REMI(config)
        self.vocab_size = vocab_size
        self.max_bar_embedding = max_bar_embedding
        
        # 【天才级防御】：无论 MidiTok 怎么改变 API，强制解析底层词表内存
        # 建立绝对的硬编码映射，彻底免疫 AttributeError 灾难
        self._str_to_id = {}
        self._id_to_str = {}
        
        if hasattr(self.tokenizer, 'vocab') and isinstance(self.tokenizer.vocab, dict):
            self._str_to_id = self.tokenizer.vocab
        elif hasattr(self.tokenizer, '_vocab_base') and isinstance(self.tokenizer._vocab_base, dict):
            self._str_to_id = self.tokenizer._vocab_base
            
        if self._str_to_id:
            self._id_to_str = {v: k for k, v in self._str_to_id.items()}

    def _get_token_string(self, token_id: int) -> str:
        """Absolute O(1) mapping from ID to Token String."""
        if token_id in self._id_to_str:
            return self._id_to_str[token_id]
        if hasattr(self.tokenizer, '__getitem__'):
            try: 
                res = self.tokenizer[token_id]
                if isinstance(res, str): return res
            except Exception: pass
        if hasattr(self.tokenizer, 'id_to_token'):
            try: return self.tokenizer.id_to_token(token_id)
            except Exception: pass
        return ""

    def _get_token_id(self, token_str: str) -> int:
        """Absolute O(1) mapping from Token String to ID."""
        if token_str in self._str_to_id:
            return self._str_to_id[token_str]
        if hasattr(self.tokenizer, '__getitem__'):
            try: 
                res = self.tokenizer[token_str]
                if isinstance(res, int): return res
            except Exception: pass
        if hasattr(self.tokenizer, 'token_to_id'):
            try: return self.tokenizer.token_to_id(token_str)
            except Exception: pass
        return 0

    def tokenize_midi(self, midi_path: str) -> List[int]:
        """
        Version-Agnostic tokenization. Will work regardless of MidiTok v2 or v3.
        Automatically unpacks underlying tensor IDs bypassing high-level objects.
        """
        try:
            # MidiTok v3 Preferred Engine
            if callable(self.tokenizer):
                tokens = self.tokenizer(midi_path)
            elif hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(midi_path)
            else:
                tokens = self.tokenizer.midi_to_tokens(midi_path)
        except Exception:
            return []
            
        if isinstance(tokens, list):
            if len(tokens) == 0: return []
            tok_seq = tokens[0]  # Take first track
        else:
            tok_seq = tokens
            
        if hasattr(tok_seq, 'ids'):
            return list(tok_seq.ids)
        elif isinstance(tok_seq, list) and len(tok_seq) > 0:
            if isinstance(tok_seq[0], int):
                return tok_seq
            elif isinstance(tok_seq[0], str):
                return [self._get_token_id(t) for t in tok_seq]
        
        return []
    
    def detokenize(self, token_ids: List[int], output_path: str):
        """Version-Agnostic MIDI Detokenization."""
        try:
            # MidiTok v3 Flow
            from miditok import TokSequence
            seq = TokSequence(ids=token_ids)
            midi = self.tokenizer.decode([seq]) if hasattr(self.tokenizer, 'decode') else self.tokenizer(seq)
            if hasattr(midi, 'dump_midi'):
                midi.dump_midi(output_path)
            elif hasattr(midi, 'dump'):
                midi.dump(output_path)
            return
        except Exception:
            pass
            
        try:
            # MidiTok v2 Flow Fallback
            tokens = [self._get_token_string(tid) for tid in token_ids]
            if hasattr(self.tokenizer, 'tokens_to_midi'):
                midi = self.tokenizer.tokens_to_midi([tokens])
            else:
                midi = self.tokenizer([tokens])
                
            if hasattr(midi, 'dump'):
                midi.dump(output_path)
            elif hasattr(midi, 'save'):
                midi.save(output_path)
        except Exception as e:
            print(f"[ERROR] Detokenization failed: {e}")
        
    def find_bar_indices(self, token_ids: List[int]) -> List[int]:
        bar_indices = []
        for i, token_id in enumerate(token_ids):
            token_str = self._get_token_string(token_id)
            if token_str.startswith("Bar"):
                bar_indices.append(i)
        return bar_indices
    
    def extract_metadata_tokens(self, token_ids: List[int], up_to_index: int) -> Dict[str, Optional[int]]:
        metadata = {'tempo': None, 'time_signature': None}
        for i in range(up_to_index - 1, -1, -1):
            token_str = self._get_token_string(token_ids[i])
            if metadata['tempo'] is None and token_str.startswith("Tempo"):
                metadata['tempo'] = token_ids[i]
            if metadata['time_signature'] is None and token_str.startswith("TimeSig"):
                metadata['time_signature'] = token_ids[i]
            if metadata['tempo'] is not None and metadata['time_signature'] is not None:
                break
        return metadata

    def is_structural_token(self, token_id: int) -> bool:
        """Safeguard: check if a token represents an atomic musical boundary."""
        token_str = self._get_token_string(token_id)
        return token_str.startswith(("Bar", "Pitch", "NoteOn", "Tempo", "TimeSig"))
    
    def get_vocab_size(self) -> int:
        if self._str_to_id: return len(self._str_to_id)
        if hasattr(self.tokenizer, '__len__'): return len(self.tokenizer)
        return 65536


def create_context_completion_pairs(
    token_ids: List[int],
    tokenizer: PianoTokenizer,
    n_context_bars: int = 4,
    n_completion_bars: int = 2,
    step: int = 1
) -> List[Dict[str, List[int]]]:
    bar_indices = tokenizer.find_bar_indices(token_ids)
    total_bars_needed = n_context_bars + n_completion_bars
    if len(bar_indices) < total_bars_needed:
        return []
    
    data_pairs = []
    for i in range(0, len(bar_indices) - total_bars_needed + 1, step):
        context_start_idx = bar_indices[i]
        completion_start_idx = bar_indices[i + n_context_bars]
        
        if (i + total_bars_needed) < len(bar_indices):
            completion_end_idx = bar_indices[i + total_bars_needed]
        else:
            completion_end_idx = len(token_ids)
        
        context_ids = token_ids[context_start_idx:completion_start_idx]
        completion_ids = token_ids[completion_start_idx:completion_end_idx]
        
        if i > 0:
            metadata = tokenizer.extract_metadata_tokens(token_ids, context_start_idx)
            prepend_tokens = []
            if metadata['tempo'] is not None:
                prepend_tokens.append(metadata['tempo'])
            if metadata['time_signature'] is not None:
                prepend_tokens.append(metadata['time_signature'])
            if prepend_tokens:
                context_ids = prepend_tokens + context_ids
        
        data_pairs.append({
            'context': context_ids,
            'completion': completion_ids
        })
    
    return data_pairs


def process_midi_directory(
    midi_dir: str,
    tokenizer: PianoTokenizer,
    n_context_bars: int = 4,
    n_completion_bars: int = 2,
    step: int = 1
) -> List[Dict[str, List[int]]]:
    all_pairs = []
    midi_files = list(Path(midi_dir).glob("**/*.mid")) + list(Path(midi_dir).glob("**/*.midi"))
    
    print(f"\n[Tokenization] Target acquired: {len(midi_files)} physical MIDI records.")
    print(f"[Notice] If you see '_wfopen_s returned: 0', ignore it. It means SUCCESS in C/C++ backend.")
    
    for idx, midi_file in enumerate(midi_files):
        try:
            token_ids = tokenizer.tokenize_midi(str(midi_file))
            if not token_ids:
                continue
            pairs = create_context_completion_pairs(
                token_ids, tokenizer, n_context_bars, n_completion_bars, step
            )
            all_pairs.extend(pairs)
        except Exception:
            # 静默过滤那些不规范的、无法被正常解析的野鸡 MIDI 文件
            continue
            
        if (idx + 1) % 50 == 0:
            print(f"  -> Sliced {idx + 1}/{len(midi_files)} files... (Tensor pairs: {len(all_pairs)})")
            
    print(f"\n[Success] Synthesized {len(all_pairs)} training pairs from {len(midi_files)} files in total.")
    return all_pairs