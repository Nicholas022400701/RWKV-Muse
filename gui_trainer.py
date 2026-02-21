import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import threading

CONFIG_FILE = "train_config.json"

class ROSATrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RWKV-8 ROSA 训练前可视化控制台")
        self.root.geometry("550x450")
        
        self.config = {
            "batch_size": 4,
            "max_seq_len": 2048,
            "lr": 1e-4,
            "epochs": 10,
            "weight_decay": 0.01,
            "grad_clip": 1.0
        }
        
        ttk.Label(root, text="🎹 钢琴音乐补全: RWKV-8 ROSA 物理参数阵列", font=("Helvetica", 14, "bold")).pack(pady=15)
        
        self.entries = {}
        frame = ttk.Frame(root)
        frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        for idx, (k, v) in enumerate(self.config.items()):
            ttk.Label(frame, text=k, font=("Helvetica", 10)).grid(row=idx, column=0, padx=10, pady=8, sticky="e")
            entry = ttk.Entry(frame, width=25, font=("Helvetica", 10))
            entry.insert(0, str(v))
            entry.grid(row=idx, column=1, padx=10, pady=8, sticky="w")
            self.entries[k] = entry
            
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=20)
        
        self.ignite_btn = ttk.Button(btn_frame, text="🔥 保存配置并点火 (Ignite!)", command=self.ignite)
        self.ignite_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)
        
    def ignite(self):
        try:
            self.config["batch_size"] = int(self.entries["batch_size"].get())
            self.config["max_seq_len"] = int(self.entries["max_seq_len"].get())
            self.config["lr"] = float(self.entries["lr"].get())
            self.config["epochs"] = int(self.entries["epochs"].get())
            self.config["weight_decay"] = float(self.entries["weight_decay"].get())
            self.config["grad_clip"] = float(self.entries["grad_clip"].get())
            
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
                
            self.ignite_btn.config(state=tk.DISABLED, text="[ 正在交接至 CUDA 核心... ]")
            print("[GUI] 参数锁定完毕，正在向 4090 推送张量执行指令...")
            
            def run_train():
                subprocess.run([sys.executable, "train_parallel.py", "--config", CONFIG_FILE])
                self.root.after(0, self.root.destroy)
                
            threading.Thread(target=run_train, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "参数解析失败，天才，请输入合法的数字！")

if __name__ == "__main__":
    root = tk.Tk()
    app = ROSATrainerGUI(root)
    root.mainloop()