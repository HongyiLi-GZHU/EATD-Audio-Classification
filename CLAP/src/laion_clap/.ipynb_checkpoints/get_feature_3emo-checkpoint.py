"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: https://arxiv.org/abs/2211.06687
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""

import numpy as np
import librosa
import torch
import laion_clap
import os

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

model = laion_clap.CLAP_Module(enable_fusion=True)
# model.load_ckpt(ckpt='/root/CLAP/630k-audioset-fusion-best.pt')
model.load_afclap_ckpt(ckpt='/root/audio-flamingo/audio-flamingo/clap_ckpt/epoch_16.pt', verbose=True)

# 设置基础目录和类别
base_dir = "/root/dataset/EATD-3classification"
categories = ["negative_out", "neutral_out", "positive_out"]
category_labels = {"negative_out": 0, "neutral_out": 1, "positive_out": 2}

# 创建保存结果的目录
output_dir = "/root/Flamingo-Embedding"
os.makedirs(output_dir, exist_ok=True)

# 存储所有嵌入和元数据
all_embeddings = []
all_filenames = []
all_labels = []

# 分批处理音频文件
batch_size = 32  # 根据GPU内存调整

for category in categories:
    print(f"\n处理类别: {category}")
    category_dir = os.path.join(base_dir, category)
    
    # 获取类别下所有.wav文件
    audio_files = []
    for root, dirs, files in os.walk(category_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    if len(audio_files) == 0:
        print(f"警告: 在 {category_dir} 中未找到任何.wav文件")
        continue
    
    # 分批处理当前类别的音频文件
    category_embeddings = []
    
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i+batch_size]
        print(f"  处理批次 {i//batch_size + 1}: {len(batch_files)} 个文件")
        
        try:
            # 获取音频嵌入
            batch_embeddings = model.get_audio_embedding_from_filelist(
                x=batch_files, 
                use_tensor=False
            )
            
            category_embeddings.append(batch_embeddings)
            
        except Exception as e:
            print(f"  处理批次时出错: {e}")
            # 尝试逐个处理文件
            for file in batch_files:
                try:
                    single_embedding = model.get_audio_embedding_from_filelist(
                        x=[file], 
                        use_tensor=False
                    )
                    category_embeddings.append(single_embedding)
                    print(f"    成功处理单个文件: {os.path.basename(file)}")
                except Exception as e_single:
                    print(f"    无法处理文件 {file}: {e_single}")
    
    if category_embeddings:
        # 合并当前类别的所有嵌入
        category_embed = np.concatenate(category_embeddings, axis=0)
        
        # 保存当前类别的嵌入
        category_output_path = os.path.join(output_dir, f"{category}_embeddings.npy")
        np.save(category_output_path, category_embed)
        print(f"  {category}类别的嵌入已保存至: {category_output_path}")
        
        # 添加到总集合中
        all_embeddings.append(category_embed)
        all_filenames.extend(audio_files)
        all_labels.extend([category_labels[category]] * len(audio_files))



