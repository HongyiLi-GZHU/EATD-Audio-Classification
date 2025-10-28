import numpy as np
import librosa
import torch
import os
from transformers import AutoFeatureExtractor, Qwen2AudioEncoder
from tqdm import tqdm


class Qwen2AudioFeatureExtractor:
    def __init__(self, model_path, target_length=10, sample_rate=16000):
        """
        初始化特征提取器

        参数:
        - model_path: 模型路径
        - target_length: 目标音频长度(秒)
        - sample_rate: 采样率(Hz)
        """
        self.model = Qwen2AudioEncoder.from_pretrained(model_path,
                                                       torch_dtype = torch.float16,
                                                       device_map = "auto")
        self.processor = AutoFeatureExtractor.from_pretrained(model_path, trust_remote_code=True)
        self.device = torch.device("cuda:0")
        self.model.to(self.device)
        self.model.eval()

        # 音频处理参数
        self.target_length = target_length  # 目标音频长度(秒)
        self.sample_rate = sample_rate  # 采样率
        self.target_samples = target_length * sample_rate  # 目标样本数

        print(f"Qwen2-Audio模型已加载到 {self.device} 设备")
        print(f"音频处理参数: 目标长度={target_length}秒, 采样率={sample_rate}Hz")

    def process_audio(self, audio):
        """处理音频数据，确保长度一致"""
        # 如果音频太短，进行填充
        if len(audio) < self.target_samples:
            padding = self.target_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')

        # 如果音频太长，进行截断
        elif len(audio) > self.target_samples:
            audio = audio[:self.target_samples]

        return audio

    def get_audio_embedding_from_file(self, file_path):
        """从单个音频文件提取嵌入"""
        try:
            # 加载音频文件
            audio, sample_rate = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=True,
                duration=self.target_length  # 限制加载长度
            )

            # 确保音频长度一致
            audio = self.process_audio(audio)

            # 处理音频输入
            inputs = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)

            # 转换attention_mask为布尔类型
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].bool()

            # 提取嵌入
            with torch.no_grad():
                outputs = self.model(inputs.input_features)
                audio_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            return audio_embedding

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return None

    def get_audio_embedding_from_filelist(self, file_list, batch_size=32):
        """从文件列表批量提取嵌入"""
        embeddings = []
        failed_files = []

        # 分批处理
        for i in tqdm(range(0, len(file_list), batch_size), desc="处理音频文件"):
            batch_files = file_list[i:i + batch_size]
            batch_embeddings = []

            for file_path in batch_files:
                embedding = self.get_audio_embedding_from_file(file_path)
                if embedding is not None:
                    batch_embeddings.append(embedding)
                else:
                    failed_files.append(file_path)

            if batch_embeddings:
                embeddings.extend(batch_embeddings)

        if embeddings:
            return np.concatenate(embeddings, axis=0), failed_files
        return None, failed_files


# 设置基础目录和类别
base_dir = "/share/workspace/lixiang/HongyiLi/HongyiLi/ML/dataset/EATD-3classification"
categories = ["negative_out", "neutral_out", "positive_out"]
category_labels = {"negative_out": 0, "neutral_out": 1, "positive_out": 2}

# 创建保存结果的目录
output_dir = "/share/workspace/lixiang/HongyiLi/HongyiLi/ML/Qwen2-Audio-Embedding"
os.makedirs(output_dir, exist_ok=True)

# 加载Qwen2-Audio模型
model_path = "/share/workspace/shared_models/01_LLM-models/Qwen2-Audio-7B"

# 初始化特征提取器，设置目标音频长度为10秒
feature_extractor = Qwen2AudioFeatureExtractor(
    model_path,
    target_length=10,  # 统一音频长度为10秒
    sample_rate=16000
)

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

    # 提取当前类别的音频嵌入
    category_embeddings, failed_files = feature_extractor.get_audio_embedding_from_filelist(
        audio_files,
        batch_size=batch_size
    )

    if category_embeddings is not None:
        # 保存当前类别的嵌入
        category_output_path = os.path.join(output_dir, f"{category}_embeddings.npy")
        np.save(category_output_path, category_embeddings)
        print(f"  {category}类别的嵌入已保存至: {category_output_path}")

        # 添加到总集合中
        all_embeddings.append(category_embeddings)
        all_filenames.extend([f for f in audio_files if f not in failed_files])
        all_labels.extend([category_labels[category]] * (len(audio_files) - len(failed_files)))

    if failed_files:
        print(f"  无法处理 {len(failed_files)} 个文件")

# 保存所有嵌入
if all_embeddings:
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(os.path.join(output_dir, "all_embeddings.npy"), all_embeddings)
    np.save(os.path.join(output_dir, "all_filenames.npy"), np.array(all_filenames))
    np.save(os.path.join(output_dir, "all_labels.npy"), np.array(all_labels))
    print(f"\n所有嵌入已保存至: {os.path.join(output_dir, 'all_embeddings.npy')}")