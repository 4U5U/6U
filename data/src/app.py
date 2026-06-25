import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

import streamlit as st
import os
import re
import requests
import tempfile
import asyncio
import edge_tts
import whisper
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

# ------------------- 语音模型缓存（ASR语音转文字） -------------------
@st.cache_resource(show_spinner="正在加载语音识别模型...")
def load_asr_model():
    """Whisper语音识别模型，全局缓存只加载一次"""
    return whisper.load_model("base")

# 语音转文字函数
def speech_to_text(audio_bytes):
    model = load_asr_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = model.transcribe(tmp_path, language="zh")
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)

# 文字转语音函数（Edge-TTS免费在线合成）
async def tts_generate(text, output_path="reply_voice.mp3"):
    voice_name = "zh-CN-YunyangNeural"
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save_sync(output_path)
    return output_path

def play_answer_voice(text):
    """生成语音并在页面自动播放"""
    try:
        audio_path = asyncio.run(tts_generate(text))
        st.audio(audio_path, autoplay=True)
        # 延时删除临时音频文件
        def remove_audio():
            import time
            time.sleep(8)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        import threading
        threading.Thread(target=remove_audio).start()
    except Exception as e:
        st.warning(f"语音播放异常：{str(e)}")

# ------------------- 原有向量库资源缓存 -------------------
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

# ========== 新增：麦克风语音输入控件 ==========
st.divider()
col1, col2 = st.columns([4,1])
with col1:
    audio_input = st.audio_input("🎤 按住麦克风说话提问校园问题")
with col2:
    st.info("语音提问模式")

# 处理语音输入逻辑
voice_text = ""
if audio_input:
    with st.spinner("正在识别语音内容..."):
        voice_text = speech_to_text(audio_input.getvalue())
        st.success(f"识别结果：{voice_text}")
        # 把识别文字赋值给提问变量
        st.session_state["temp_input"] = voice_text

# 聊天输入框，支持侧边快捷填充 + 语音填充
input_text = st.chat_input("请输入你的校园问题...")
# 侧边快捷提问 / 语音提问赋值
if "temp_input" in st.session_state and st.session_state["temp_input"]:
    input_text = st.session_state["temp_input"]
    del st.session_state["temp_input"]

# 处理用户提问（文字/语音共用一套问答逻辑）
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
        # 自动朗读回答语音
        play_answer_voice(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
