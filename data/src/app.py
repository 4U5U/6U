import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import streamlit as st
import os
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa   # 导入工具函数

load_dotenv()

# ------------------- 缓存资源 -------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True}
    )

@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    return Chroma(persist_directory="./vector_db", embedding_function=embeddings)

embeddings = load_embeddings()
vector_db = load_vector_db()

APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 请在 .env 文件中设置 SPARK_APIPASSWORD")
    st.stop()

# ------------------- RAG 问答 -------------------
def rag_retrieve_answer(question):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt_text = RAG_PROMPT.format(context=context, question=question)

    url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APIPASSWORD}"
    }
    payload = {
        "model": "spark-x",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API 错误：{resp.status_code} - {resp.text}"
    except Exception as e:
        return f"⚠️ 请求异常：{e}"

# ------------------- 智能体路由 -------------------
def agent_answer(question):
    if re.search(r'第.*周|校历|本周|几周', question):
        return get_current_week()
    if re.search(r'绩点|GPA|平均分', question):
        nums = re.findall(r'\d+', question)
        if nums:
            return calculate_gpa(','.join(nums))
        else:
            return "请提供您的各科分数，例如：85,90,78"
    return rag_retrieve_answer(question)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="校园百事通", page_icon="🏫")
st.title("🏫 校园生活百事通助手")
st.markdown("我可以回答校园问题，还能查询校历周数和计算绩点！")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入你的校园问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            answer = agent_answer(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})