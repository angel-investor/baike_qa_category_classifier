import ollama
import requests

# TODO Python中使用ollama API的方式


# 使用Python直接调用ollama：方式一
def dm01():
    """
    如果是访问本地服务器直接使用ollama即可
    如果是访问远程服务器，则需要使用ollama.Client创建一个新的对象并指定主机地址: new_ollama = ollama.Client(url)
    """
    # 获取用户要问的问题
    pro = '给我讲一个笑话'

    # 访问本地服务器,获取响应
    res = ollama.chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": pro}]
    )
    print('res-->', res)  # 整个对象
    # 只获取响应内容的两种方式
    print(res.message.content)
    # print(res['message']['content'])


# 使用Python直接调用ollama：方式二
def dm02():
    from ollama import Client
    client = Client(host='http://127.0.0.1:11434')
    response = client.chat(model='gpt-oss:20b', messages=[
        {
            'role': 'user',
            'content': '为什么天空是蓝⾊的？',
        },
    ])
    print(response['message']['content'])


# 使用Python直接调用ollama：方式三
def dm03():
    stream = ollama.chat(
        model='gpt-oss:20b',
        messages=[{'role': 'user', 'content': '你为什么那么帅？'}],
        stream=True,
    )
    # print("stream-->", stream) # 只是个对象
    for chunk in stream:
        # print("chunk-->", chunk)
        print(chunk['message']['content'], end='', flush=True)


# 使用requests库调用ollama
def dm04():
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gpt-oss:20b",  # 模型选择
        "options": {
            "temperature": 0.  # 为0表示不让模型⾃由发挥，输出结果相对较固定，> 0的话，输出的结果会⽐较放⻜⾃我
    },
    "stream": False,  # 流式输出
    "messages": [{
        "role": "system",
        "content": "你是谁？"
    }]  # 对话列表
    }
    response = requests.post(url='http://127.0.0.1:11434/api/chat', json=data, headers=headers, timeout=60)
    print(response.json()['message']['content'])


# 使用langchain框架调用ollama
def dm05():
    from langchain_community.llms import Ollama
    # 如果⾃⼰本地系统有ollama服务，可以省略base_url
    llm = Ollama(base_url="http://127.0.0.1:11434",
                 model="gpt-oss:20b", temperature=0)
    res = llm.invoke("你是谁")
    print(res)


if __name__ == '__main__':
    # dm01()
    dm02()
    # dm03()
    # dm04()
    # dm05()