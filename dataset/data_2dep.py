import os
import shutil

def reorganize_eatd_dataset(original_dataset_path, new_dataset_path):
    """
    重整EATD数据集用于抑郁/非抑郁二分类任务。

    Args:
        original_dataset_path (str): 原始EATD数据集的根目录路径。
        new_dataset_path (str): 新组织的数据集将要存放的根目录路径。
    """

    # 定义新数据集的目录结构
    splits = ['train', 'val']  # 对应原始的 't' 和 'v' 开头文件夹
    categories = ['depressed', 'non_depressed']

    # 创建新目录
    for split in splits:
        for category in categories:
            dir_path = os.path.join(new_dataset_path, split, category)
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")

    # 映射关系：原始文件夹前缀 -> 新数据集中的分割名称
    split_mapping = {'t': 'train', 'v': 'val'}

    # 遍历原始数据集目录
    for item in os.listdir(original_dataset_path):
        item_path = os.path.join(original_dataset_path, item)
        
        # 检查是否是目录并且以 't' 或 'v' 开头
        if os.path.isdir(item_path) and item[0] in split_mapping:
            current_split = split_mapping[item[0]]  # 确定是训练集还是验证集
            
            # 构建label.txt文件的路径
            label_file_path = os.path.join(item_path, 'new_label.txt')
            
            # 读取SDS评分
            if not os.path.isfile(label_file_path):
                print(f"警告: 在 {item_path} 中未找到new_label.txt，跳过。")
                continue
                
            with open(label_file_path, 'r') as f:
                sds_score = float(f.read().strip())
            
            # 根据SDS评分确定类别（抑郁/非抑郁）
            current_category = 'depressed' if sds_score >= 50 else 'non_depressed'
            
            # 定义三段语音文件
            audio_files = ['negative_out.wav', 'neutral_out.wav', 'positive_out.wav']
            
            # 复制并重命名每个语音文件到新目录
            for audio_file in audio_files:
                source_audio_path = os.path.join(item_path, audio_file)
                
                if os.path.isfile(source_audio_path):
                    # 生成新的文件名，避免重复（例如：t001_negative_out.wav）
                    new_filename = f"{item}_{audio_file}"
                    destination_audio_path = os.path.join(new_dataset_path, current_split, current_category, new_filename)
                    
                    # 复制文件
                    shutil.copy2(source_audio_path, destination_audio_path)
                    print(f"已复制: {source_audio_path} -> {destination_audio_path}")
                else:
                    print(f"警告: 在 {item_path} 中未找到音频文件 {audio_file}，跳过。")

    print("数据集重整完成！")

# 使用示例
if __name__ == "__main__":
    # 请替换为您的实际路径
    original_path = "./EATD-Corpus"    # 原始数据集路径
    new_path = "./EATD-2classification"  # 新数据集路径

    reorganize_eatd_dataset(original_path, new_path)