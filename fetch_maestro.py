"""
Genius Protocol: MAESTRO Dataset Harvester
Directly rips high-fidelity piano performance data from Google infrastructure.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
ZIP_PATH = "maestro_v3.zip"
RAW_DIR = "data/raw_midi"

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        sys.stdout.write(f"\r[Genius Protocol] Extracting data from Google Servers... {percent:.1f}%")
        sys.stdout.flush()

def harvest():
    print("========================================================")
    print("[Genius Protocol] MAESTRO Dataset Harvester Initiated")
    print("========================================================")
    
    os.makedirs(RAW_DIR, exist_ok=True)
    
    if not os.path.exists(ZIP_PATH):
        print(f"[*] Target locked: {MAESTRO_URL}")
        urllib.request.urlretrieve(MAESTRO_URL, ZIP_PATH, download_progress)
        print("\n[+] Quantum Transfer Complete.")
    else:
        print("[+] Archive already exists in local matrix.")

    print("[*] Ripping archive and flattening directory structure...")
    extracted = 0
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for file_info in z.infolist():
            if file_info.filename.lower().endswith(('.mid', '.midi')):
                # 暴力降维：无视原有的年份文件夹树，直接把所有 MIDI 铺平
                filename = os.path.basename(file_info.filename)
                target = os.path.join(RAW_DIR, filename)
                
                with z.open(file_info) as source, open(target, "wb") as dest:
                    shutil.copyfileobj(source, dest)
                
                extracted += 1
                if extracted % 100 == 0:
                    print(f"  -> Extracted {extracted} master-class performances...")
                    
    print(f"[+] Harvested {extracted} performance-grade MIDI files into {RAW_DIR}/")
    
    # 阅后即焚，清理战场
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("[*] Temporary archive erased.")
        
    print("=== [Data Harvest Complete] ===")

if __name__ == "__main__":
    harvest()