import numpy as np
import librosa
import torch
import laion_clap
import os
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class EATDFeatureExtractor:
    def __init__(self, model_path, target_sr=48000):
        """
        初始化EATD特征提取器
        
        Args:
            model_path: CLAP模型权重路径
            target_sr: 目标采样率，CLAP模型通常使用48000Hz
        """
        self.target_sr = target_sr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化CLAP模型
        self.model = laion_clap.CLAP_Module(enable_fusion=True)
        self.model.load_ckpt(ckpt=model_path, verbose=True)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✅ CLAP模型加载成功，使用设备: {self.device}")
    
    def load_and_preprocess_audio(self, audio_path, duration=10.0):
        """
        加载并预处理音频文件
        
        Args:
            audio_path: 音频文件路径
            duration: 音频时长（秒），超过将截断，不足将填充
            
        Returns:
            processed_audio: 预处理后的音频数据
        """
        try:
            # 加载音频文件
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # 计算目标长度
            target_length = int(duration * self.target_sr)
            current_length = len(audio)
            
            if current_length > target_length:
                # 截断音频
                audio = audio[:target_length]
            elif current_length < target_length:
                # 填充音频
                padding = target_length - current_length
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # 确保音频在[-1, 1]范围内
            audio = np.clip(audio, -1.0, 1.0)
            
            return audio.astype('float32')
            
        except Exception as e:
            print(f"❌ 音频加载失败: {audio_path}, 错误: {e}")
            return None
    
    def extract_features(self, audio_data):
        """
        提取音频特征
        
        Args:
            audio_data: 预处理后的音频数据
            
        Returns:
            features: 音频特征向量
        """
        if audio_data is None:
            return None
            
        try:
            # 将音频数据转换为模型期望的格式
            # CLAP模型期望形状为 (batch_size, audio_length)
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model.get_audio_embedding_from_data(
                    audio_tensor, 
                    use_tensor=True
                )
            
            # 转换为numpy数组并返回
            return features.cpu().numpy().squeeze()
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return None
    
    def process_dataset(self, dataset_path, output_path, emotion_types=None):
        """
        批量处理整个数据集
        
        Args:
            dataset_path: 重整后数据集的路径
            output_path: 特征保存路径
            emotion_types: 要处理的情感类型列表
        """
        if emotion_types is None:
            emotion_types = ['negative_out', 'neutral_out', 'positive_out']
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 存储所有特征和标签
        all_features = []
        all_labels = []
        all_file_paths = []
        all_emotion_types = []
        all_splits = []  # 训练集或测试集标识
        
        # 遍历数据集目录结构
        splits = ['train', 'val']  # 对应重整后的train和val目录
        
        for split in splits:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                print(f"⚠️ 跳过不存在的目录: {split_path}")
                continue
                
            for category in ['depressed', 'non_depressed']:
                category_path = os.path.join(split_path, category)
                if not os.path.exists(category_path):
                    print(f"⚠️ 跳过不存在的目录: {category_path}")
                    continue
                
                print(f"\n🔍 处理: {split}/{category}")
                
                # 获取该类别下的所有音频文件
                audio_files = []
                for emotion in emotion_types:
                    pattern = f"*{emotion}.wav"
                    emotion_files = [f for f in os.listdir(category_path) if f.endswith(f'{emotion}.wav')]
                    audio_files.extend([(f, emotion) for f in emotion_files])
                
                if not audio_files:
                    print(f"⚠️ 在 {category_path} 中未找到音频文件")
                    continue
                
                # 处理每个音频文件
                for audio_file, emotion in tqdm(audio_files, desc=f"处理{category}"):
                    audio_path = os.path.join(category_path, audio_file)
                    
                    # 加载和预处理音频
                    audio_data = self.load_and_preprocess_audio(audio_path)
                    if audio_data is None:
                        continue
                    
                    # 提取特征
                    featureforSave = []
                    features = self.extract_features(audio_data)
                    featureforSave = np.array(features)

                    
                    if features is not None:
                        all_features.append(features)
                        all_file_paths.append(audio_path)
                        all_emotion_types.append(emotion)
                        all_splits.append(split)
                        
                        # 根据目录结构确定标签
                        label = 1 if category == 'depressed' else 0
                        all_labels.append(label)

                        # feature_filename = os.path.splitext(audio_file)[0] + '.npy'
                        
                        feature_file_path = os.path.join(output_path, split, category)
                                
                        # 保存特征文件
                        np.save(feature_file_path, features)
                    
                
                    
        
        # 转换为numpy数组
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        # 保存特征和元数据
        self.save_features(all_features, all_labels, all_file_paths, 
                          all_emotion_types, all_splits, output_path)
        
        return all_features, all_labels
    
    def save_features(self, features, labels, file_paths, emotion_types, splits, output_path):
        """
        保存特征和元数据
        """
        # 保存特征数组
        np.save(os.path.join(output_path, 'audio_features.npy'), features)
        np.save(os.path.join(output_path, 'audio_labels.npy'), labels)
        
        # 保存元数据为CSV
        metadata = {
            'file_path': file_paths,
            'label': labels,
            'emotion_type': emotion_types,
            'split': splits
        }
        
        df_metadata = pd.DataFrame(metadata)
        df_metadata.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
        
        # 保存特征统计信息
        feature_stats = {
            'total_samples': len(features),
            'feature_dim': features.shape[1] if len(features.shape) > 1 else features.shape[0],
            'depressed_count': np.sum(labels == 1),
            'non_depressed_count': np.sum(labels == 0),
            'train_count': np.sum(np.array(splits) == 'train'),
            'val_count': np.sum(np.array(splits) == 'val')
        }
        
        # 打印统计信息
        print(f"\n📊 特征提取完成！")
        print(f"📁 总样本数: {feature_stats['total_samples']}")
        print(f"🔢 特征维度: {feature_stats['feature_dim']}")
        print(f"😔 抑郁样本: {feature_stats['depressed_count']}")
        print(f"😊 非抑郁样本: {feature_stats['non_depressed_count']}")
        print(f"🏋️ 训练集样本: {feature_stats['train_count']}")
        print(f"🧪 验证集样本: {feature_stats['val_count']}")
        print(f"💾 特征保存路径: {output_path}")
    
    def load_saved_features(self, feature_path):
        """
        加载已保存的特征
        """
        features = np.load(os.path.join(feature_path, 'audio_features.npy'))
        labels = np.load(os.path.join(feature_path, 'audio_labels.npy'))
        metadata = pd.read_csv(os.path.join(feature_path, 'metadata.csv'))
        
        return features, labels, metadata

# 使用示例
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "/root/model/630k-audioset-fusion-best.pt"
    DATASET_PATH = "/root/dataset/EATD-2classification"  # 替换为您的重整后数据集路径
    OUTPUT_PATH = "/root/CLAP-Embedding/features"      # 替换为您希望保存特征的路径
    
    # 初始化特征提取器
    extractor = EATDFeatureExtractor(model_path=MODEL_PATH)
    
    # 处理整个数据集
    features, labels = extractor.process_dataset(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH
    )
    
    print("✅ 特征提取流程完成！")