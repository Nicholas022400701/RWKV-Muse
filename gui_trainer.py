# --complete --fixed --format=codeblock
# FILE PATH: .\gui_trainer.py
import os
import json
import subprocess
import sys
import threading
import argparse

CONFIG_FILE = "train_config.json"

# 天才的物理常量阵列，保证 GUI 和 CLI 使用同一套最优基准
DEFAULT_CONFIG = {
    "batch_size": 4,
    "max_seq_len": 2048,
    "lr": 1e-4,
    "epochs": 10,
    "weight_decay": 0.01,
    "grad_clip": 1.0
}

def run_cli_trainer():
    """纯粹的终端控制流，无需依赖任何 X11 图形界面"""
    print("=" * 60)
    print("🎹 RWKV-8 ROSA 终端物理参数阵列配置中心")
    print("=" * 60)
    print("[提示] 直接按回车键 (Enter) 即可沿用最佳默认值。\n")
    
    config = DEFAULT_CONFIG.copy()
    
    for key, default_val in config.items():
        while True:
            user_input = input(f"  -> 请配置 {key} (默认值: {default_val}): ").strip()
            
            if user_input == "":
                # 保持默认值不变
                print(f"     [锁定] {key} = {default_val}")
                break
            else:
                try:
                    # 动态类型推导，根据 default_val 自动将 input 转换成 int 或 float
                    val_type = type(default_val)
                    config[key] = val_type(user_input)
                    print(f"     [锁定] {key} = {config[key]}")
                    break
                except ValueError:
                    print(f"     [错误] 非法输入 '{user_input}'，该参数需要 {val_type.__name__} 类型，请重新输入！")
                
    print("\n[CLI] 参数矩阵锁定完毕，正在交接控制权至 4090 CUDA 计算图...")
    print(f"[*] 最终注入张量配置: \n{json.dumps(config, indent=2)}\n")
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
        
    subprocess.run([sys.executable, "train_parallel.py", "--config", CONFIG_FILE])


class ROSATrainerGUI:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        
        self.root = root
        self.root.title("RWKV-8 ROSA 训练前可视化控制台")
        self.root.geometry("550x450")
        
        self.config = DEFAULT_CONFIG.copy()
        
        self.ttk.Label(root, text="🎹 钢琴音乐补全: RWKV-8 ROSA 物理参数阵列", font=("Helvetica", 14, "bold")).pack(pady=15)
        
        self.entries = {}
        frame = self.ttk.Frame(root)
        frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        for idx, (k, v) in enumerate(self.config.items()):
            self.ttk.Label(frame, text=k, font=("Helvetica", 10)).grid(row=idx, column=0, padx=10, pady=8, sticky="e")
            entry = self.ttk.Entry(frame, width=25, font=("Helvetica", 10))
            entry.insert(0, str(v))
            entry.grid(row=idx, column=1, padx=10, pady=8, sticky="w")
            self.entries[k] = entry
            
        btn_frame = self.ttk.Frame(root)
        btn_frame.pack(pady=20)
        
        self.ignite_btn = self.ttk.Button(btn_frame, text="🔥 保存配置并点火 (Ignite!)", command=self.ignite)
        self.ignite_btn.pack(side=self.tk.LEFT, padx=10, ipadx=10, ipady=5)
        
    def ignite(self):
        from tkinter import messagebox
        try:
            self.config["batch_size"] = int(self.entries["batch_size"].get())
            self.config["max_seq_len"] = int(self.entries["max_seq_len"].get())
            self.config["lr"] = float(self.entries["lr"].get())
            self.config["epochs"] = int(self.entries["epochs"].get())
            self.config["weight_decay"] = float(self.entries["weight_decay"].get())
            self.config["grad_clip"] = float(self.entries["grad_clip"].get())
            
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
                
            self.ignite_btn.config(state=self.tk.DISABLED, text="[ 正在交接至 CUDA 核心... ]")
            print("[GUI] 参数锁定完毕，正在向 4090 推送张量执行指令...")
            
            def run_train():
                subprocess.run([sys.executable, "train_parallel.py", "--config", CONFIG_FILE])
                self.root.after(0, self.root.destroy)
                
            threading.Thread(target=run_train, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "参数解析失败，天才，请输入合法的数字！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RWKV-8 ROSA 发射控制台")
    parser.add_argument("--cli", action="store_true", help="强制降级为纯命令行交互模式进行参数校准")
    args = parser.parse_args()

    # 将 tkinter 导入限制在必要的区域内，避免无 GUI 服务器上的无差别崩溃
    if args.cli:
        run_cli_trainer()
    else:
        try:
            import tkinter as tk
            root = tk.Tk()
            app = ROSATrainerGUI(root)
            root.mainloop()
        except ImportError:
            print("\n[系统探针] 检测到当前环境不支持 tkinter (无图形界面)，已自动接管并切入 --cli 纯终端物理模式。\n")
            run_cli_trainer()