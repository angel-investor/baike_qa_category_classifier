# 测试调用api接口
import requests

# 准备url
url = 'http://127.0.0.1:8009/predict'

try:
    # 发送POST请求
    text = input("请输入要预测的文本：")
    r = requests.post(url, json={'text': text})
    # print(r.json())
    print("预测结果:", r.json()['pred_class'])

except Exception as e:
    print("Error occurred:", e)
