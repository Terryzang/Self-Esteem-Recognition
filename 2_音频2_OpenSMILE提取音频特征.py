import opensmile
import pandas as pd
import os

# 初始化 OpenSMILE，使用 IS10_paralinguistics.conf 进行特征提取
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals  # 选择提取的特征级别
)

# 定义批量处理函数
def batch_process_wav(input_folder, output_csv):
    """
    遍历 input_folder 中所有 WAV 文件，提取 OpenSMILE 语音特征，并保存为 CSV。
    第一列为被试 ID。
    参数:
    - input_folder: str, 包含 WAV 文件的文件夹路径
    - output_csv: str, 输出的 CSV 文件路径
    """
    data = []  # 存储所有特征
    n = 1

    # 遍历文件夹中的所有 WAV 文件
    for filename in sorted(os.listdir(input_folder)):
        print(n)
        if filename.endswith(".wav"):
            # 提取 ID（前三位）
            participant_id = filename[:3]

            # 获取完整文件路径
            wav_path = os.path.join(input_folder, filename)

            # 处理音频文件，提取特征
            features = smile.process_file(wav_path)

            # 添加 ID 列
            features.insert(0, "ID", participant_id)

            # 存入列表
            data.append(features)
            n += 1

    # 合并所有数据
    df = pd.concat(data, ignore_index=True)

    # 保存为 CSV
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. Saved to {output_csv}")

input_folder = f"E:"  # WAV 文件所在文件夹
output_csv = f"E:"  # 输出 CSV 文件
batch_process_wav(input_folder, output_csv)
