"""
Genius Protocol: Dynamic G1 Weights Harvester
"""
import os
import urllib.request
import json
import sys

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        sys.stdout.write(f"\r[Genius Protocol] Extracting G1 Weights... {percent:.2f}% ({downloaded/(1024*1024):.1f}MB)")
        sys.stdout.flush()

os.makedirs('./models', exist_ok=True)
TARGET_FILE = "./models/rwkv_430m.pth"

if __name__ == "__main__":
    if not os.path.exists(TARGET_FILE):
        print("="*60)
        print(" RWKV7-G1 'GooseOne' Quantum Harvester")
        print("="*60)
        print("[*] Intercepting HuggingFace API to locate the latest G1 0.4B model...")
        
        # 智能切换国内加速镜像 API
        api_url = "https://hf-mirror.com/api/models/BlinkDL/rwkv7-g1/tree/main"
        try:
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                files = json.loads(response.read().decode())
                
            # 过滤出 0.4b 的 pth 文件
            pth_files = [f['path'] for f in files if f['path'].endswith('.pth') and '0.4b' in f['path'].lower()]
            if not pth_files:
                raise Exception("No 0.4B .pth file found in the repository.")
            
            # 按文件名排序（时间戳），取最新编译的基座
            latest_file = sorted(pth_files)[-1]
            download_url = f"https://hf-mirror.com/BlinkDL/rwkv7-g1/resolve/main/{latest_file}"
            
            print(f"[*] Target locked: {latest_file}")
            print("[*] Initiating high-speed quantum transfer...")
            
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(download_url, TARGET_FILE, download_progress)
            print("\n[+] Transfer Complete. G1 Weights are physically mounted.")
            
        except Exception as e:
            print(f"\n[ERROR] Network anomaly or API block: {e}")
            print("如果下载失败，请手动前往：")
            print("https://huggingface.co/BlinkDL/rwkv7-g1/tree/main")
            print("下载最新日期的 0.4b 权重文件，重命名为 rwkv_430m.pth 后放入 models 文件夹。")
    else:
        print("[+] Local G1 weight matrix already exists.")