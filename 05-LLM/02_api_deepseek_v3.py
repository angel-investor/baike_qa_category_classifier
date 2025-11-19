from openai import OpenAI


def dm01():
    client = OpenAI(
        api_key="sk-2e0c12c9200645fc91d86816faf01004",
        base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)


# 调用第三方模型（远程），这里以DeepSeek为例。
def dm02(prompt):
    # 创建客户端
    client = OpenAI(api_key="sk-2e0c12c9200645fc91d86816faf01004", base_url="https://api.deepseek.com")
    # 创建会话
    response = client.chat.completions.create(
        # 模型deepseek-chat是deepseek-v3
        # 模型deepseek-reasoner是deepseek-r1
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个ai助手"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    # 打印结果
    print(response.choices[0].message.content)


if __name__ == '__main__':
    # dm01()
    dm02("给我讲一个笑话")
    # 调用大模型来实现分类任务
    # dm02("体验2D巅峰 倚天屠龙记十大创新概览")