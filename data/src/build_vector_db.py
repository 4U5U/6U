import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 配置 HuggingFace 国内镜像，解决访问超时
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载数据
df = pd.read_csv('data/campus_data.csv')

# 使用免费嵌入模型，增加参数避免联网校验卡住
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    # 禁止初始化时远程拉取配置，防止超时
    cache_folder="./hf_model_cache",
)

# 创建向量库
texts = df['answer'].tolist()
metadatas = df[['id', 'category', 'question']].to_dict('records')

print(f"开始构建向量库，总待处理文本数量：{len(texts)}")
vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory='./vector_db'
)
vector_db.persist()

print(f"✅ 向量库构建完成！已存入{len(texts)}条记录")
print(f"✅ 向量库存放路径：{os.path.abspath('./vector_db')}")