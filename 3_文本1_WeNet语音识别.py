import subprocess
import os
from tqdm import tqdm


def transcribe_wav(wenet_path, wav_path):
    # 保存当前目录
    original_cwd = os.getcwd()

    # 更改工作目录到 Wenet 目录
    os.chdir(wenet_path)

    # 构建命令行命令
    command = f"wenet --language chinese \"{wav_path}\""

    # 执行命令并获取输出
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 恢复原始工作目录
    os.chdir(original_cwd)

    # # 返回标准输出
    return result.stdout

def batch_transcribe(wenet_path, data_folder_path):
    # 获取所有 WAV 文件
    wav_files = [f for f in os.listdir(data_folder_path) if f.endswith('.wav')]

    # 使用 tqdm 显示进度条
    for filename in tqdm(wav_files, desc="Processing", unit="file"):
        wav_path = os.path.join(data_folder_path, filename)
        output = transcribe_wav(wenet_path, wav_path)

        # 创建同名的 TXT 文件
        txt_path = os.path.join(data_folder_path, filename.replace('.wav', '.txt'))
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(output)



# Wenet 目录
wenet_path = 'c:\\'

# 数据文件夹路径
data_folder_path = 'E:\\'

# 批量处理
batch_transcribe(wenet_path, data_folder_path)
