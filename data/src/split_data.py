# src/split_data.py
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 读取csv里所有问答，拼接成完整长文本
all_text = ""
with open("./data/campus_data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row["question"]
        a = row["answer"]
        all_text += f"问题：{q} 回答：{a}\n"

long_text = all_text

# 2. 配置切分参数
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

# 3. 切分文本
chunks = text_splitter.split_text(long_text)

# 4. 输出结果
print(f"切分片段总数：{len(chunks)}")
for idx, chunk in enumerate(chunks):
    print(f"\n【分段{idx+1}】\n{chunk}")