import os

def genius_merge(output_name="merged_knowledge.txt"):
    # 获取脚本自身的绝对路径
    self_path = os.path.abspath(__file__)
    
    # 获取输出文件的绝对路径 (关键修改点 1)
    # 必须用绝对路径比较，防止 './merged.txt' 和 'merged.txt' 这种字符串不匹配的问题
    output_path = os.path.abspath(output_name)

    # 目标集合，O(1) 查找
    targets = {'.txt', '.md','.py','.cu','.cpp','.bat','.sh'}

    print(f"正在扫描并合并文件到: {output_path} ...")

    with open(output_name, 'w', encoding='utf-8') as outfile:
        # os.walk 是处理文件树最高效的方式
        for root, _, files in os.walk('.'):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext not in targets:
                    continue
                
                full_path = os.path.join(root, file)
                abs_path = os.path.abspath(full_path) # 获取当前文件的绝对路径

                # 排除脚本自身
                if abs_path == self_path:
                    continue
                
                # 排除正在生成的输出文件 (关键修改点 2)
                if abs_path == output_path:
                    # print(f"跳过输出文件: {file}") # 可选：调试用
                    continue

                # 写入分隔线和元数据
                header = f"\n{'='*60}\nFILE PATH: {full_path}\n{'='*60}\n"
                outfile.write(header)

                try:
                    # errors='replace' 防止编码错误中断
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"\n[ERROR READING FILE: {e}]\n")
                
                outfile.write("\n")

if __name__ == "__main__":
    genius_merge()
    print("处理完毕。")
