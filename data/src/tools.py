# src/tools.py
from datetime import datetime

def get_current_week():
    """获取当前是第几周（校历）"""
    today = datetime.now()
    week_num = (today - datetime(2026, 3, 2)).days // 7 + 1
    return f"现在是第{week_num}周"

def calculate_gpa(scores_str):
    """计算绩点，输入格式：'85,90,78'"""
    scores = [int(x) for x in scores_str.split(',')]
    total = 0
    for s in scores:
        if s >= 90:
            total += 4.0
        elif s >= 80:
            total += 3.0
        elif s >= 70:
            total += 2.0
        elif s >= 60:
            total += 1.0
        else:
            total += 0
    gpa = total / len(scores)
    return f"您的平均绩点是：{gpa:.2f}"

# 程序入口，运行文件时自动执行
if __name__ == "__main__":
    print(get_current_week())
    print(calculate_gpa("85,90,78"))