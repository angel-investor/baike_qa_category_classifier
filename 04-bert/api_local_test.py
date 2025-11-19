# 测试调用api接口
import requests

# 准备url
url = 'http://127.0.0.1:8010/predict'

try:
    # 获取用户问题
    text = input("请输入文本内容：")
    # 发送POST请求
    r = requests.post(url, json={'text': text})
    print(r.json())

except Exception as e:
    print("Error occurred:", e)
