# src/test_retrieve.py
# 配置 HuggingFace 国内镜像，解决访问超时
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 初始化和构建时一致的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh"
)

# 加载本地持久化的向量库
vector_db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

def search_knowledge(query):
    results = vector_db.similarity_search(query, k=3)
    for r in results:
        print(f"相关度: {r.metadata}")
        print(f"内容: {r.page_content}\n")
    return results
# 测试
search_knowledge("我发烧了怎么办？")            