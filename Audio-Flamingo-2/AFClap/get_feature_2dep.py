"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: 链接
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""

import numpy as np
import librosa
import torch
import my_laion_clap.CLAP.src.laion_clap as laion_clap
import os
from tqdm import tqdm

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

def process_long_audio(model, audio_path, max_duration=10.0, sr=16000):
    """
    处理长音频文件，将其分割成片段并提取嵌入
    """
    try:
        # 加载音频文件
        audio, sr = librosa.load(audio_path, sr=sr)
        duration = len(audio) / sr
        
        # 如果音频长度超过最大时长，则分割处理
        if duration > max_duration:
            segment_length = int(max_duration * sr)
            segments = []
            
            # 计算需要分割的段数
            num_segments = int(np.ceil(len(audio) / segment_length))
            
            # 分割音频
            for i in range(num_segments):
                start = i * segment_length
                end = min((i + 1) * segment_length, len(audio))
                segment = audio[start:end]
                segments.append(segment)
            
            # 处理每个片段
            segment_embeddings = []
            for segment in segments:
                # 使用模型处理单个片段
                segment_embedding = model.get_audio_embedding_from_data(
                    [segment],
                    sr=sr
                )
                segment_embeddings.append(segment_embedding)
            
            # 平均所有片段的嵌入
            avg_embedding = np.mean(segment_embeddings, axis=0)
            return avg_embedding
        else:
            # 直接处理短音频
            embedding = model.get_audio_embedding_from_data(
                [audio],
                sr=sr
            )
            return embedding
            
    except Exception as e:
        print(f"处理音频 {audio_path} 时出错: {e}")
        return None

# 加载模型
model = laion_clap.CLAP_Module(enable_fusion=True, 
        amodel='HTSAT-afclap',
        tmodel='t5')
model.load_afclap_ckpt(ckpt='/root/audio-flamingo/audio-flamingo/clap_ckpt/epoch_16.pt', verbose=True)

# 设置基础目录和类别
base_dir = "/root/dataset/EATD-2classification"
spilts= ["train", "val"]
categories = ["depressed", "non_depressed"]
category_labels = {"depressed": 0, "non_depressed": 1}

# 创建保存结果的目录
output_dir = "/root/EATD-Repo/Audio-Flamingo-2/Flamingo-Embedding/2-Dep-Classification"
os.makedirs(output_dir, exist_ok=True)

# 存储所有嵌入和元数据
all_embeddings = []
all_filenames = []
all_labels = []

# 处理每个类别
for spilt in spilts:
    spilt_dir = os.path.join(base_dir, spilt)
    for category in categories:
        print(f"\n处理类别: {category}")
        category_dir = os.path.join(spilt_dir, category)
        
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
        
        # 处理当前类别的所有音频文件
        category_embeddings = []
        successful_files = []
        
        for audio_file in tqdm(audio_files, desc=f"处理 {category} 文件"):
            try:
                # 处理音频文件（包括长音频分段处理）
                embedding = process_long_audio(model, audio_file, max_duration=10.0, sr=16000)
                
                if embedding is not None:
                    category_embeddings.append(embedding)
                    successful_files.append(audio_file)
                    print(f"成功处理: {os.path.basename(audio_file)}")
                else:
                    print(f"处理失败: {os.path.basename(audio_file)}")
                    
            except Exception as e:
                print(f"处理文件 {audio_file} 时出错: {e}")
        
        if category_embeddings:
            # 合并当前类别的所有嵌入
            category_embed = np.vstack(category_embeddings)
            
            # 保存当前类别的嵌入
            category_output = os.path.join(output_dir, spilt, category)
            category_output_path = os.path.join(category_output, f"{category}.npy")
            np.save(category_output_path, category_embed)
            print(f"  {category}类别的嵌入已保存至: {category_output_path}")
            
            # 添加到总集合中
            all_embeddings.append(category_embed)
            all_filenames.extend(successful_files)
            all_labels.extend([category_labels[category]] * len(successful_files))

# 保存所有嵌入和元数据
if all_embeddings:
    all_embeddings_concat = np.vstack(all_embeddings)
    np.save(os.path.join(output_dir, "all_embeddings.npy"), all_embeddings_concat)
    print(f"所有嵌入已保存至: {os.path.join(output_dir, 'all_embeddings.npy')}")
    
    # 保存文件名和标签
    with open(os.path.join(output_dir, "filenames.txt"), "w") as f:
        for filename in all_filenames:
            f.write(f"{filename}\n")
    
    np.save(os.path.join(output_dir, "labels.npy"), np.array(all_labels))
    print(f"文件名和标签已保存")
else:
    print("未成功处理任何音频文件")