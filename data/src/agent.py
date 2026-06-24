import re
from tools import get_current_week, calculate_gpa
from rag import rag_answer

# 全局对话存储
conversation_history = []

def agent_chat(user_input):
    # 意图1：查询校历周数
    if "周" in user_input and ("几" in user_input or "校历" in user_input):
        return get_current_week()
    # 意图2：计算绩点
    if "绩点" in user_input or "GPA" in user_input:
        scores = re.findall(r"\d+", user_input)
        if scores:
            return calculate_gpa(",".join(scores))
        else:
            return "请输入各科分数，格式示例：85,90,78"
    # 默认走RAG校园问答
    return rag_answer(user_input)

# 带对话记忆封装
def chat_with_memory(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})
    # 只保留最近5轮对话
    recent_ctx = conversation_history[-5:]
    reply = agent_chat(user_input)
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

# 本地测试
if __name__ == "__main__":
    print(chat_with_memory("现在第几周"))
    print(chat_with_memory("帮我算绩点85,90,78"))
    print(chat_with_memory("食堂开饭时间"))