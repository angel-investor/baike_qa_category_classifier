import requests
# 定义接口地址
url = 'http://127.0.0.1:8008/predict'

try:
    # 测试调用api接口
    # 获取用户录入的百科数据
    text = input("请输入你的百科数据：")
    response = requests.post(url, json={'text': text})
    print('response-->', response.json())

except Exception as e:
    print("Error occurred:", e)

