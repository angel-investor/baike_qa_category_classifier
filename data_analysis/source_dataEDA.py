import json
import os
import random
from collections import Counter
import time
from tqdm import tqdm
from config import Config

conf = Config()
train_path = conf.train_path
valid_path = conf.valid_path
test_path = conf.test_path
"""
整合了数据EDA操作和类别排序操作
"""

# ========================================
# 按 value 降序排序 JSON 的独立方法
# ========================================
def sort_json_by_value(in_path, out_path):
    """
    读取一个 JSON 文件（必须是 {key: value} 格式），
    按 value 降序排序并写回新文件。
    """
    print(f"\n开始排序文件：{in_path}")

    # 读取 JSON
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 按 value 降序排序
    sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

    # 打印检查
    print(json.dumps(sorted_data, ensure_ascii=False, indent=2))

    # 写回文件
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)

    print("排序并写入完成：", out_path)
    return sorted_data


# ========================================
# 数据统计 EDA
# ========================================
def data_eda(path, out_path):
    print('开始读取数据...')
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), desc="读取数据", unit="行"):
            data.append(json.loads(line))

    # 总数据量
    print('len(data)-->', len(data))

    # 统计类别分布
    category_counter = Counter()

    for item in data:
        cate = item.get('category', None)
        if cate:
            category_counter[cate] += 1

    print("类别总数：", len(category_counter))
    print("所有类别：")
    for k, v in category_counter.most_common():
        print(f"{k}: {v}")

    # 写入未排序的 category 统计结果
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(category_counter, f, ensure_ascii=False, indent=2)

    print("保存成功:", out_path)

    # 文本长度分析
    keys = ['title', 'desc', 'answer']

    for key in keys:
        lengths = [len(item.get(key, '')) for item in data]

        count = len(lengths)
        mean_len = sum(lengths) / count
        max_len = max(lengths)
        min_len = min(lengths)
        std_len = (sum((l - mean_len) ** 2 for l in lengths) / count) ** 0.5

        print(f"\n=== {key} 长度统计 ===")
        print(f"条目数量：{count}")
        print(f"平均长度：{mean_len:.2f} 字符")
        print(f"长度标准差：{std_len:.2f} 字符")
        print(f"最大长度：{max_len} 字符")
        print(f"最小长度：{min_len} 字符")


# ========================================
# 主函数
# ========================================
if __name__ == '__main__':
    start_time = time.time()

    # 清洗后的训练/验证文件
    train_clean_path = '../data/missing_clean_train.json'
    valid_clean_path = '../data/missing_clean_valid.json'

    # 类别输出（未排序）
    out_path = '../data/output_data/missing_distribution_clean_train.json'

    # 执行 EDA
    data_eda(train_clean_path, out_path)
    # data_eda(valid_clean_path, out_path)

    # 再对统计结果排序
    sort_json_by_value(
        in_path=out_path,
        out_path='../data/output_data/missing_distribution_clean_train_sort.json'
    )

    print("运行时间：%.2f秒" % (time.time() - start_time))