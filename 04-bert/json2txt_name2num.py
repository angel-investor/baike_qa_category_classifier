import json
import random
from tqdm import tqdm


# =============================== 基础功能 ===============================

def load_category_map(category_file):
    """读取类别文件，构造成 {类别名称: 数字ID}"""
    cat2id = {}
    with open(category_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            cat = line.strip()
            if cat:
                cat2id[cat] = idx
    return cat2id


def load_raw_json(json_path):
    """加载原始 JSON 数据"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# =============================== 随机采样 ===============================

def random_sample(data, sample_size=None, sample_ratio=None):
    """
    随机采样
    sample_size: 指定数量
    sample_ratio: 指定比例 (0 < ratio <= 1)
    """
    total = len(data)

    if not sample_size and not sample_ratio:
        return data

    if sample_size:
        sample_size = min(sample_size, total)
        print(f"按数量采样：{sample_size}/{total}")
        return random.sample(data, sample_size)

    if sample_ratio:
        if not (0 < sample_ratio <= 1):
            raise ValueError("sample_ratio 必须在 (0, 1] 范围内")
        sample_size = int(total * sample_ratio)
        print(f"按比例采样：{sample_ratio} → {sample_size}/{total}")
        return random.sample(data, sample_size)

    return data


# =============================== 文本清洗函数 ===============================

def clean_text(text):
    """清洗文本，避免 \t 等非法字符造成训练集错误"""
    if not text:
        return ""

    text = text.replace("\t", " ")       # ★ 关键修复：去除制表符
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\u3000", "")    # 全角空格
    text = text.replace("", "")         # 异常字符
    text = text.strip()

    return text


# =============================== JSON → 样本 ===============================

def convert_to_samples(data, cat2id):
    """将 JSON 数据转换成 (text, category_id)"""
    processed = []

    for item in tqdm(data, desc="Converting Samples"):
        raw_text = item.get("text", "").strip()
        cat = item.get("category", "").strip()

        if not raw_text or not cat:
            continue
        if cat not in cat2id:
            continue

        text = clean_text(raw_text)  # ★ 加入清洗
        if not text:
            continue

        processed.append((text, cat2id[cat]))

    return processed


# =============================== 保存 txt ===============================

def save_txt(data, save_path):
    """
    保存 txt： text \t category_id
    已保证 text 内不含 tab，不会破坏 split("\t")
    """
    with open(save_path, "w", encoding="utf-8") as f:
        for text, cid in data:
            text = text.replace("\t", " ")  # 双重保险
            f.write(f"{text}\t{cid}\n")


# =============================== 主流程 ===============================

def process_json_file(input_json, output_txt, category_file, sample_size=None, sample_ratio=None):
    """
    处理任意 JSON 文件 → txt 文件

    参数：
        input_json: 输入 JSON 文件（包含 category/text）
        output_txt: 输出 txt 文件路径
        category_file: 类别列表文件
        sample_size: 随机采样数量
        sample_ratio: 随机采样比例（0-1）
    """

    print(f"\n=== 处理文件：{input_json} ===")

    # 加载类别映射
    cat2id = load_category_map(category_file)

    # 加载原始数据
    data = load_raw_json(input_json)
    print(f"原始样本数：{len(data)}")

    # 随机采样
    data = random_sample(data, sample_size=sample_size, sample_ratio=sample_ratio)
    print(f"采样后样本数：{len(data)}")

    # 转换格式
    samples = convert_to_samples(data, cat2id)
    print(f"有效样本数：{len(samples)}")

    # 保存
    save_txt(samples, output_txt)

    print(f"已输出到：{output_txt}\n")
    return samples


# =============================== 执行入口 ===============================

if __name__ == "__main__":
    category_file = "../data/category_list.txt"

    print("================处理训练集==============")
    process_json_file(
        input_json="../data/final_train.json",
        output_txt="../data/train.txt",
        category_file=category_file,
        sample_ratio=0.2
    )

    print("================处理验证集==============")
    process_json_file(
        input_json="../data/final_valid.json",
        output_txt="../data/valid.txt",
        category_file=category_file,
        sample_ratio=None
    )

    print("================处理测试集==============")
    process_json_file(
        input_json="../data/final_test.json",
        output_txt="../data/test.txt",
        category_file=category_file,
        sample_ratio=None
    )
