from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd

# 确保 CUDA 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 读取Excel文件
data_path = f'E:\\'
data = pd.read_excel(data_path)

# 加载 MiniRBT-H256 模型和分词器
bert_path = 'E:\\MiniRBT_h256_pt'
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert = BertModel.from_pretrained(bert_path, return_dict=True).to(device)

# 存储所有特征
features_list = []

# 逐行处理数据
for index, row in data.iterrows():
    text = row['text']
    number = row['number']
    self_esteem = row['self-esteem']

    # Tokenize 并发送到 GPU/CPU
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 计算 CLS 特征
    with torch.no_grad():
        outputs = bert(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()  # 扁平化处理(256,)

    # 组合所有信息
    features_list.append([number] + cls_embedding.tolist() + [self_esteem])

    # 进度监控
    if (index + 1) % 20 == 0:
        print(f"已处理 {index + 1} / {len(data)} 条文本")

# 转换为 DataFrame
columns = ["number"] + [f"dim_{i+1}" for i in range(cls_embedding.shape[0])] + ["self-esteem"]
features_df = pd.DataFrame(features_list, columns=columns)

# 保存到Excel
output_path = f'E:\\'
features_df.to_csv(output_path, index=False)

print(f"特征提取完成，结果已保存至 {output_path}")

