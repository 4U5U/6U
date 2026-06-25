import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

import streamlit as st
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

# ===================== 全局初始化 =====================
load_dotenv()
st.set_page_config(
    page_title="校园百事通助手",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 自定义样式（大幅美化+三栏布局适配） =====================
def inject_global_css():
    css = """
    <style>
    /* 全局背景 */
    .stApp {
        background: linear-gradient(145deg, #f8faff, #edf2fb);
    }
    /* 顶部标题栏容器 */
    .top-header {
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        padding: 22px 35px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.18);
    }
    /* 侧边栏容器 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 12px rgba(0,0,0,0.06);
        padding: 15px 8px;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #2563eb;
    }
    /* 侧边按钮样式 */
    [data-testid="stSidebar"] button {
        width: 100%;
        border-radius: 10px;
        margin: 4px 0;
        border: 1px solid #dbeafe;
        background: #f8fafc;
        transition: 0.2s;
    }
    [data-testid="stSidebar"] button:hover {
        background: #dbeafe;
        border-color: #2563eb;
    }
    /* 右侧信息卡片 */
    .right-info-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.07);
        margin-bottom: 18px;
    }
    /* AI消息居左气泡 */
    div[data-testid="stChatMessage"][aria-label="assistant"] .stMarkdown {
        background: #ffffff;
        border-radius: 18px 18px 18px 5px;
        padding: 14px 20px;
        box-shadow: 0 2px 9px rgba(0,0,0,0.07);
        max-width: 83%;
    }
    /* 用户消息居右蓝色气泡 */
    div[data-testid="stChatMessage"][aria-label="user"] {
        display: flex;
        flex-direction: row-reverse;
    }
    div[data-testid="stChatMessage"][aria-label="user"] .stMarkdown {
        background: #2563eb;
        color: white;
        border-radius: 18px 18px 5px 18px;
        padding: 14px 20px;
        box-shadow: 0 2px 9px rgba(37, 99, 235, 0.2);
        max-width: 83%;
    }
    /* 聊天输入框美化 */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px;
        border: 1px solid #bfdbfe;
        padding: 13px 18px;
    }
    hr {
        border-color: #e2e8f0 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_global_css()

# ===================== 资源缓存函数（封装优化） =====================
@st.cache_resource(show_spinner="正在加载文本向量化模型...")
def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="正在加载校园知识库向量库...")
def create_vector_database():
    embed_model = create_embedding_model()
    db_folder = "./vector_db"
    if not os.path.exists(db_folder):
        st.warning(f"向量库目录 {db_folder} 不存在，请先导入文档生成知识库！")
    return Chroma(persist_directory=db_folder, embedding_function=embed_model)

# 初始化全局资源
vector_db = create_vector_database()
SPARK_KEY = os.getenv("SPARK_APIPASSWORD")
if not SPARK_KEY:
    st.error("❌ 缺少环境变量：请在项目根目录 .env 配置 SPARK_APIPASSWORD 星火密钥")
    st.stop()

# ===================== 业务逻辑函数（封装解耦） =====================
def rag_query(question: str) -> str:
    """RAG知识库问答核心逻辑"""
    try:
        docs = vector_db.similarity_search(question, k=3)
        if not docs:
            return "📭 知识库未查询到相关内容，暂时无法解答该问题，你可以咨询教务处相关老师。"
        
        context_text = "\n\n=====分割线=====\n\n".join([doc.page_content.strip() for doc in docs])
        prompt = RAG_PROMPT.format(context=context_text, question=question)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SPARK_KEY}"
        }
        payload = {
            "model": "spark-x",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024
        }
        resp = requests.post(
            url="https://spark-api-open.xf-yun.com/x2/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ 接口调用异常\n状态码：{resp.status_code}\n详情：{resp.text[:300]}"

    except requests.exceptions.Timeout:
        return "⏱️ 请求超时，服务器响应缓慢，请重新提问！"
    except Exception as err:
        return f"⚠️ 问答异常：{str(err)}"

def route_question(question: str) -> str:
    """意图分发路由：周数查询 / GPA计算 / RAG问答"""
    week_reg = r'第几周|第\d+周|本周|校历|现在几周|教学周'
    gpa_reg = r'绩点|GPA|平均分|算分|成绩换算'

    if re.search(week_reg, question):
        return get_current_week()
    if re.search(gpa_reg, question):
        score_list = re.findall(r'\d+', question)
        if score_list:
            return calculate_gpa(','.join(score_list))
        else:
            return """📝 绩点计算使用说明
请在问题中带上你的各科分数，示例：
帮我算绩点：88,76,92,65"""
    return rag_query(question)

# ===================== 会话状态初始化 =====================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是校园百事通，有任何校园问题、想查教学周、计算绩点都可以直接问我~"}
        ]
    if "temp_input" not in st.session_state:
        st.session_state.temp_input = ""

init_session_state()

# ===================== 页面布局：三栏布局【左 + 中 + 右】 =====================
# 顶部通栏标题
st.markdown("""
<div class="top-header">
    <h1 style="margin:0;font-size:28px;">🏫 校园生活百事通助手</h1>
    <p style="margin:8px 0 0 0;opacity:0.9;">基于本地校园知识库 + RAG大模型智能问答 | 教学周查询 · GPA绩点换算 · 校园问题咨询</p>
</div>
""", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2.2, 0.8])

# ========== 左侧边栏：功能控制面板 ==========
with st.sidebar:
    st.header("⚙️ 功能控制面板")
    st.divider()

    st.subheader("✨ 三大核心能力")
    st.markdown("""
    1. 📚 校园知识库问答
    > 规章制度、宿舍、选课、奖学金、社团咨询

    2. 📅 教学周自动查询
    > 一键获取当前学期教学周

    3. 📊 百分制GPA绩点换算
    > 批量成绩计算平均绩点
    """)
    st.divider()

    st.subheader("💡 快捷提问模板")
    quick_questions = [
        "现在是第几教学周？",
        "帮我计算绩点 90,82,75,60",
        "学校奖学金申请条件是什么？",
        "奖学金评定需要什么基础条件?"
    ]
    for q in quick_questions:
        if st.button(q):
            st.session_state.temp_input = q

    st.divider()
    if st.button("🗑️ 清空全部对话记录", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.temp_input = ""
        st.rerun()

# ========== 中间主栏：聊天对话区域 ==========
with col_center:
    # 渲染历史对话
    for msg in st.session_state.messages:
        avatar_icon = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar_icon):
            st.markdown(msg["content"])

    # 输入框逻辑
    user_input = st.chat_input("请输入你的校园问题...")
    if st.session_state.temp_input:
        user_input = st.session_state.temp_input
        st.session_state.temp_input = ""

    # 处理提问
    if user_input:
        # 存入会话
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # 生成回答
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI正在检索知识库并思考答案..."):
                answer = route_question(user_input)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ========== 右侧辅助信息面板（新增，布局改动极强） ==========
with col_right:
    st.markdown('<div class="right-info-card">', unsafe_allow_html=True)
    st.subheader("📌 项目说明")
    st.markdown("""
    - 技术架构：LangChain + Chroma向量库 + BGE嵌入模型 + 讯飞星火大模型
    - 方案模式：RAG检索增强生成，基于校内文档精准答疑
    - 适用场景：校园制度咨询、教务查询、成绩绩点换算
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="right-info-card">', unsafe_allow_html=True)
    st.subheader("📝 使用小贴士")
    st.markdown("""
    1. 直接打字提问即可对话
    2. 侧边点击快捷语句一键提问
    3. 算绩点需要带上具体数字
    4. 内容查不到可咨询教务处老师
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="right-info-card">', unsafe_allow_html=True)
    st.subheader("🟢 运行状态")
    st.success("✅ 向量库加载完成")
    st.success("✅ 大模型接口就绪")
    st.info("会话条数：" + str(len(st.session_state.messages)))
    st.markdown('</div>', unsafe_allow_html=True)
