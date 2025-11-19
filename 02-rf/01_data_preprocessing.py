import json
import random
import jieba
import pandas as pd
from config import Config
from tqdm import tqdm

# 加载配置类对象
conf = Config()

def process_json_data(json_path, processed_datapath, sample_size=None, sample_ratio=None):
    # 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"原始数据条数：{total}")

    # =========================
    #  采样逻辑
    # =========================
    if sample_size is not None:
        sample_size = min(sample_size, total)
        data = random.sample(data, sample_size)
        print(f"已随机采样 {sample_size} 条数据")

    elif sample_ratio is not None:
        sample_size = int(total * sample_ratio)
        data = random.sample(data, sample_size)
        print(f"按比例 {sample_ratio} 采样得到 {sample_size} 条数据")

    # 转为 DataFrame
    df_data = pd.DataFrame(data)

    # 重命名列
    df_data = df_data.rename(columns={"category": "label"})

    print("df_data 示例：")
    print(df_data.head())

    # 分词 + tqdm
    words_list = []
    for text in tqdm(df_data["text"], desc="分词中"):
        words_list.append(" ".join(jieba.lcut(text)[:50]))
    df_data["words"] = words_list

    # 保存 TSV
    df_data.to_csv(processed_datapath, sep='\t', index=False, header=True)
    print(f"已保存到：{processed_datapath}")


if __name__ == "__main__":
    # 例如：每个数据集随机抽 20%
    process_json_data(conf.train_datapath, conf.process_train_datapath, sample_ratio=0.2)
    process_json_data(conf.test_datapath, conf.process_test_datapath)
    process_json_data(conf.dev_datapath, conf.process_dev_datapath)

    # 或者 固定采样数量：
    # process_json_data(conf.train_datapath, conf.process_train_datapath, sample_size=200000)


    # 展示处理好的数据
    # print(pd.read_csv(conf.process_train_datapath, sep='\t', encoding='utf-8')['label'].head())
    # print(pd.read_csv(conf.process_train_datapath, sep='\t', encoding='utf-8')['text'].head())
    # print(pd.read_csv(conf.process_train_datapath, sep='\t', encoding='utf-8')['words'].head())
    # print(pd.read_csv(conf.process_test_datapath, sep='\t', encoding='utf-8').head())
    # print(pd.read_csv(conf.process_dev_datapath, sep='\t', encoding='utf-8').head())

