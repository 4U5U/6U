import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

import streamlit as st
import os
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

# 加载环境变量
load_dotenv()

# ------------------- 页面基础配置 -------------------
st.set_page_config(
    page_title="校园百事通",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- 缓存资源（增加加载提示） -------------------
@st.cache_resource(show_spinner="正在加载文本向量化模型...")
def load_embeddings():
    """加载BGE中文嵌入模型，全局缓存只加载一次"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="正在加载校园知识库向量库...")
def load_vector_db():
    """加载本地Chroma向量数据库"""
    embeddings = load_embeddings()
    db_path = "./vector_db"
    # 检测向量库文件夹是否存在
    if not os.path.exists(db_path):
        st.warning(f"向量库目录 {db_path} 不存在，请先导入校园文档生成知识库！")
    return Chroma(persist_directory=db_path, embedding_function=embeddings)

# 初始化全局向量资源
embeddings = load_embeddings()
vector_db = load_vector_db()

# 校验星火API密钥
APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 环境变量缺失！请在项目根目录 .env 文件中配置 SPARK_APIPASSWORD 星火接口密钥")
    st.stop()

# ------------------- RAG知识库问答模块（增强检索空值判断） -------------------
def rag_retrieve_answer(question):
    """
    校园知识库检索问答流程
    1. 向量相似度检索3条最相关文档
    2. 拼接上下文传入自定义RAG提示词
    3. 调用讯飞星火大模型生成答案
    """
    try:
        # 相似度检索
        docs = vector_db.similarity_search(question, k=3)
        if len(docs) == 0:
            return "📭 知识库未查询到相关内容，暂时无法解答该问题，你可以咨询教务处相关老师。"
        
        # 拼接参考上下文
        context = "\n\n=====分割线=====\n\n".join([doc.page_content.strip() for doc in docs])
        prompt_text = RAG_PROMPT.format(context=context, question=question)

        # 星火API请求参数
        url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {APIPASSWORD}"
        }
        payload = {
            "model": "spark-x",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.3,
            "max_tokens": 1024
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ 大模型接口调用失败\n状态码：{resp.status_code}\n返回信息：{resp.text[:300]}"

    except requests.exceptions.Timeout:
        return "⏱️ 请求超时，服务器响应缓慢，请重新提问！"
    except Exception as e:
        return f"⚠️ 问答流程出现未知异常：{str(e)}"

# ------------------- 工具路由智能分发模块（优化正则匹配） -------------------
def agent_answer(question):
    """
    智能路由分发：识别用户意图，匹配对应工具或RAG问答
    1. 周数查询意图匹配
    2. GPA绩点计算意图匹配
    3. 其余问题走知识库RAG问答
    """
    week_pattern = r'第几周|第\d+周|本周|校历|现在几周|教学周'
    gpa_pattern = r'绩点|GPA|平均分|算分|成绩换算'

    # 判断周数查询
    if re.search(week_pattern, question):
        return get_current_week()
    
    # 判断绩点计算
    if re.search(gpa_pattern, question):
        nums = re.findall(r'\d+', question)
        if nums:
            return calculate_gpa(','.join(nums))
        else:
            return """📝 绩点计算使用说明
请在问题中带上你的各科分数，示例：
帮我算绩点：88,76,92,65"""
    
    # 默认知识库问答
    return rag_retrieve_answer(question)

# ------------------- 侧边栏功能面板（新增丰富交互） -------------------
with st.sidebar:
    st.header("⚙️ 功能面板")
    st.divider()
    # 功能介绍
    st.subheader("✨ 三大核心能力")
    st.markdown("""
    1. 📚 校园知识库问答
    > 规章制度、宿舍、选课、奖学金、社团等校内问题
    
    2. 📅 教学周自动查询
    > 自动获取当前学期第几教学周
    
    3. 📊 百分制GPA绩点换算
    > 批量输入成绩计算平均绩点
    """)
    st.divider()
    # 快捷示例提问
    st.subheader("💡 快速提问示例")
    sample_q = [
        "现在是第几教学周？",
        "帮我计算绩点 90,82,75,60",
        "学校奖学金申请条件是什么？",
        "奖学金评定需要什么基础条件?"
    ]
    for q in sample_q:
        if st.button(q):
            st.session_state["temp_input"] = q
    st.divider()
    # 清空对话按钮
    if st.button("🗑️ 清空全部对话记录", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# ------------------- 主页面聊天UI美化 -------------------
st.title("🏫 校园生活百事通助手")
st.markdown("""
> 基于本地校园知识库 + 大模型RAG智能问答，兼顾周数查询、绩点计算，一站式解决校园全部疑问
""")
st.divider()

# 初始化对话记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是校园百事通，有任何校园问题、想查教学周、计算绩点都可以直接问我~"}
    ]

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🤖"):
        st.markdown(msg["content"])

# 聊天输入框，支持侧边快捷填充
input_text = st.chat_input("请输入你的校园问题...")
# 侧边快捷提问赋值
if "temp_input" in st.session_state and st.session_state["temp_input"]:
    input_text = st.session_state["temp_input"]
    del st.session_state["temp_input"]

# 处理用户提问
if input_text:
    # 保存用户消息
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user", avatar="👤"):
        st.markdown(input_text)

    # 生成回复
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("AI正在检索知识库并思考答案..."):
            res = agent_answer(input_text)
        st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
