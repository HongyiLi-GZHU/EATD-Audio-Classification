import os
import shutil
from pathlib import Path

# 定义原始数据集路径和新数据集路径
original_data_path = "./EATD-Corpus"
new_data_path = "./EATD-3classification"

# 定义类别列表
classes = ['negative_out', 'neutral_out', 'positive_out']

# 为每个类别创建子目录
for class_name in classes:
    os.makedirs(os.path.join(new_data_path, class_name), exist_ok=True)

# 遍历原始数据目录
for root, dirs, files in os.walk(original_data_path):
    for file in files:
        if file in ['negative_out.wav', 'neutral_out.wav', 'positive_out.wav']:
            # 获取类别名
            class_name = file.replace('.wav', '')
            # 生成新的文件名，例如：t_1_negative_out.wav
            folder_name = os.path.basename(root) # 例如 t_1, v_1
            new_filename = f"{folder_name}_{file}"
            # 源文件路径
            src_path = os.path.join(root, file)
            # 目标文件路径
            dst_path = os.path.join(new_data_path, class_name, new_filename)
            # 复制文件
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} -> {dst_path}")