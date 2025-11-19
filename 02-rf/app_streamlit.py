import streamlit as st
import requests
import time

# streamlit创建页面
# 设置标题
st.title("投满分分类项目")
st.write("这是一个投满分分类项目")
# 获取用户输入的文本
text = st.text_input("请输入要查询分类的文本:")

# 后台发送请求
# 准备url
url = 'http://127.0.0.1:8008/predict'
if st.button("基于随机森林模型获取分类！"):
    start_time = time.time()
    try:
        # 发送请求
        response = requests.post(url, json={'text': text})
        print('response-->', response.json())
        # 计算耗时
        estimated_time = (time.time() - start_time) * 1000
        st.write("耗时：", estimated_time, "ms")
        # 显示结果到页面
        st.write("预测结果", response.json()["pred_class"])
    except Exception as e:
        print("Error occurred:", e)
