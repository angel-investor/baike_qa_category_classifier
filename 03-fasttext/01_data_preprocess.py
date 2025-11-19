import json
import jieba
import random
from tqdm import tqdm
from config import Config

# 加载配置
conf = Config()


def process_json_to_fasttext(
        json_path,
        processed_datapath,
        is_char=True,
        sample_size=None,
        sample_ratio=None,
        max_words=None,
):
    """
    将 JSON 数据处理成 fastText 格式
    """

    # ===============================
    #  读取 JSON
    # ===============================
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"原始数据条数：{total}")

    # ===============================
    #  采样逻辑（与你提供脚本一致）
    # ===============================
    if sample_size is not None:
        sample_size = min(sample_size, total)
        data = random.sample(data, sample_size)
        print(f"已随机采样 {sample_size} 条数据")

    elif sample_ratio is not None:
        sample_size = int(total * sample_ratio)
        data = random.sample(data, sample_size)
        print(f"按比例 {sample_ratio} 采样得到 {sample_size} 条数据")

    print("示例：", data[0])

    # ===============================
    #  转换为 fastText 格式
    # ===============================
    with open(processed_datapath, "w", encoding="utf-8") as fw:

        for item in tqdm(data, desc="处理中"):

            text = item["text"]
            label_str = item["category"]  # fastText 需要字符串

            # --- 分词 ---
            if is_char:
                text_split = " ".join(list(text))
            else:
                words = jieba.lcut(text)
                if max_words is not None:  # 支持截断（你脚本里是 50）
                    words = words[:max_words]
                text_split = " ".join(words)

            # --- fastText 格式 ---
            ft_line = f"__label__{label_str} {text_split}\n"

            fw.write(ft_line)

    print(f"已保存到：{processed_datapath}")


if __name__ == "__main__":
    # 字符级别
    process_json_to_fasttext(conf.train_datapath, conf.process_train_datapath_char, is_char=True, sample_ratio=0.999)
    process_json_to_fasttext(conf.test_datapath, conf.process_test_datapath_char, is_char=True)
    process_json_to_fasttext(conf.dev_datapath, conf.process_dev_datapath_char, is_char=True)

    # 分词级别（限制最多50词，可自行改）
    process_json_to_fasttext(conf.train_datapath, conf.process_train_datapath_word, is_char=False, max_words=50, sample_ratio=0.999)
    process_json_to_fasttext(conf.test_datapath, conf.process_test_datapath_word, is_char=False, max_words=50)
    process_json_to_fasttext(conf.dev_datapath, conf.process_dev_datapath_word, is_char=False, max_words=50)
