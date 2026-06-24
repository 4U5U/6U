# 修复sqlite版本过低问题，放在代码最顶部
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import os
# 国内镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_TIMEOUT"] = "120"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from prompt_templates import RAG_PROMPT

load_dotenv()

# 初始化嵌入和向量库（保持不变）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"trust_remote_code": True}
)
vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)

# 填入你讯飞后台真实完整 APIPassword
APIPASSWORD = "SVsCwxkAmfhyPjcRqzUm:obsbwUQSfLSmsDOMQZXS"

def rag_answer(question):
    # 1. 向量检索
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    
    # 2. 拼接提示词
    prompt_text = RAG_PROMPT.format(context=context, question=question)
    
    # Ultra-32K 官方HTTP接口地址
    url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APIPASSWORD}"
    }
    payload = {
        # ==========关键修改：ultra → 4.0Ultra ==========
        "model": "4.0Ultra",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"HTTP 请求失败（{response.status_code}）：{response.text}"
    except Exception as e:
        return f"请求异常：{e}"

# 测试
if __name__ == "__main__":
    print(rag_answer("怎么请病假？"))
    print(rag_answer("奖学金要多少绩点？"))
    print(rag_answer("宿舍灯坏了找谁？"))
    print(rag_answer("一卡通丢了怎么办？"))
    print(rag_answer("选错课能退吗？"))