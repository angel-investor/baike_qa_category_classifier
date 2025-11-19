import streamlit as st
import requests
import time

# streamlit创建页面
st.title("投满分分类项目")
st.write("这是一个投满分分类项目")
text = st.text_input("请输入要查询的文本")
# 发送请求
url = 'http://127.0.0.1:8009/predict'
if st.button("基于fasttext获取分类！"):
    start_time = time.time()
    try:
        # 发送 POST 请求
        r = requests.post(url, json={'text': text})
        print(r.json())
        # 显示结果到页面
        st.write("预测结果：", r.json()["pred_class"])

        # 计算耗时
        estimated_time = (time.time() - start_time) * 1000
        st.write("耗时：", estimated_time, "ms")

    except Exception as e:
        print("Error occurred:", e)
