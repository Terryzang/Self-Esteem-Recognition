import os
import subprocess
from tqdm import tqdm

# 定义OpenFace可执行文件路径
openface_exe = "D:\\OpenFace_2.2.0_win_x64\\FeatureExtraction.exe"

# 定义输入和输出文件夹路径
input_folder = "E:\\"
output_folder = "E:\\"

# 创建输出文件夹，如果不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有MP4视频文件
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.mp4'):
        input_video = os.path.join(input_folder, filename)
        output_dir = os.path.join(output_folder, os.path.splitext(filename)[0])

        # 创建每个视频的输出文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 构建命令行
        command = [
            openface_exe,
            "-f", input_video,
            "-out_dir", output_dir,
            "-au_static",
        ]

        # 运行命令
        subprocess.run(command, check=True)

print("所有视频处理完成。")
