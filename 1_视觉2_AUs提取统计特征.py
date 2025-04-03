import numpy as np
import pandas as pd
import os
from tqdm import tqdm


# 计算特征
def calculate_features(file_path):
    data = pd.read_csv(file_path)
    # 初始化
    features = {}
    # 计算AUr的四阶统计特征
    for au in [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
               ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r',
               ' AU45_r']:
        features[f'{au}_mean'] = data[au].mean()
        features[f'{au}_std'] = data[au].std()
        features[f'{au}_skew'] = data[au].skew()
        features[f'{au}_kurt'] = data[au].kurt()
    return features


# 批处理所有子文件夹中的CSV文件
def process_all_files(input_root, output_file):
    all_features = []

    # 计算文件总数
    total_files = 211
    file_count = 0

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                subject_id = file[:3]
                features = calculate_features(file_path)
                features['subject_id'] = subject_id
                all_features.append(features)

                file_count += 1
                # 更新进度条
                tqdm.write(f'Processing file {file_count}/{total_files}: {subject_id}')

    df = pd.DataFrame(all_features)
    df.to_csv(output_file, index=False)

input_root = "E:"
output_file = "E:"

process_all_files(input_root, output_file)
print(f"Processed features saved to {output_file}")

