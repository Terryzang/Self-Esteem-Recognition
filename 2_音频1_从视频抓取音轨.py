import os
import subprocess

# 定义函数
def extract_wav_from_mp4(directory,out_directory):
    # 检查目录是否存在
    if not os.path.exists(directory):
        return "Directory not found."

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            # 构造文件路径
            mp4_path = os.path.join(directory, filename)
            wav_path = os.path.join(out_directory, filename.replace(".mp4", ".wav"))

            # 使用 FFmpeg 提取音频，并转换为 WAV（PCM 16-bit, 16kHz, 单声道）
            command = [
                "ffmpeg",
                "-i", mp4_path,  # 输入 MP4 文件
                "-vn",  # 忽略视频流
                "-acodec", "pcm_s16le",  # 设置音频编码为 PCM 16-bit
                "-ar", "16000",  # 采样率调整为 16kHz
                "-ac", "1",  # 转换为单声道
                wav_path  # 输出 WAV 文件
            ]

            # 执行命令
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return "Extraction complete."

# 使用函数
directory_path = f"E:\\" # 替换为目录路径
out_directory = f'E:\\'
extract_wav_from_mp4(directory_path,out_directory)