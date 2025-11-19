import json
import random
from collections import defaultdict

def stratified_split(
    input_path: str,
    train_out_path: str,
    test_out_path: str,
    test_size: int = 50000
):
    """
    按类别比例抽取 test_size 条数据作为测试集，
    剩余数据作为训练集，并保存到指定文件。
    """

    # =====================
    # 1. 读取数据
    # =====================
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"加载数据完成，总样本数：{total}")

    # =====================
    # 2. 按类别分桶
    # =====================
    buckets = defaultdict(list)
    for item in data:
        buckets[item["category"]].append(item)

    # =====================
    # 3. 按比例抽取测试集
    # =====================
    test_data = []
    remain_data = []

    for cat, samples in buckets.items():
        cat_count = len(samples)

        # 该类别分得的测试集数量
        cat_test_num = int(cat_count / total * test_size)
        cat_test_num = min(cat_test_num, cat_count)

        # 随机抽取
        selected = random.sample(samples, cat_test_num)
        test_data.extend(selected)

        # 剩余样本
        for item in samples:
            if item not in selected:
                remain_data.append(item)

    # =====================
    # 4. 补齐不足的测试样本（由于四舍五入）
    # =====================
    if len(test_data) < test_size:
        shortage = test_size - len(test_data)
        print(f"⚠️ 测试集少 {shortage} 条，随机从训练集中补齐...")

        extra = random.sample(remain_data, shortage)
        test_data.extend(extra)

        # 从训练集中剔除补充的部分
        extra_ids = set(id(x) for x in extra)
        remain_data = [x for x in remain_data if id(x) not in extra_ids]

    print(f"最终测试集数量：{len(test_data)}")
    print(f"最终训练集数量：{len(remain_data)}")

    # =====================
    # 5. 保存文件
    # =====================
    with open(train_out_path, "w", encoding="utf-8") as f:
        json.dump(remain_data, f, ensure_ascii=False, indent=2)

    with open(test_out_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("文件已保存：")
    print("训练集：", train_out_path)
    print("测试集：", test_out_path)

if __name__ == '__main__':
    stratified_split(
        input_path="../data/pure_train.json",
        train_out_path="../data/final_train.json",
        test_out_path="../data/final_test.json",
        test_size=50000
    )