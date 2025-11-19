import json
import os
import random
from config import Config

conf = Config()
train_path = conf.train_path
dev_path = conf.dev_path
test_path = conf.test_path


# 去除特殊符号函数
def clean_text(s):
    return (
        s.replace('\r', '\n')  # 去掉回车符
        .replace('', '')  # 去掉杂乱全角空格
        .replace('\u3000', '')  # 去掉换行符
        .replace('\x7f', '')  # \x7f 是一个 控制字符（不可见字符）
        .strip()
    )


# 处理原始数据中的特殊符号
def data_clean():
    # 从原始数据集读取数据
    data = []
    with open(train_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # 先解析JSON
            json_data = json.loads(line)
            # 再清理文本
            cleaned_desc = clean_text(json_data['desc'])
            cleaned_desc = clean_text(json_data['answer'])
            # 更新清理后的描述
            json_data['desc'] = cleaned_desc
            json_data['answer'] = cleaned_desc

            data.append(json_data)

            print("data--> ", data[idx])
            print("data['qid']-->", data[idx]['qid'])
            print("data['category']-->", data[idx]['category'])
            print("data['title']-->", data[idx]['title'])
            print("data['desc']-->", data[idx]['desc'])
            print("data['answer']-->", data[idx]['answer'])
            print("====" * 20)

            if idx == 5:
                break


if __name__ == '__main__':
    data_clean()
