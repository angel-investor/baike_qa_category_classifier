import json


def count_missing_fields(json_path):
    """
    统计 JSON 文件中 qid / category / title / desc / answer 五个字段缺失的数量
    （按行读取，每行一个样本格式）
    """

    # 需要检查的 key
    required_keys = ["qid", "category", "title", "desc", "answer"]

    # 统计缺失情况
    missing_count = {key: 0 for key in required_keys}
    total = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                item = json.loads(line)
            except:
                # 如果某行格式异常，可以记录一下
                continue

            for key in required_keys:
                if key not in item or item[key] in ["", None]:
                    missing_count[key] += 1

    print(f"样本总数: {total}")
    print("各字段缺失数量:")
    for k, v in missing_count.items():
        print(f"{k}: {v}")

    return missing_count


def clean_json_lines(input_path, output_path):
    """
    删除缺失字段（qid/category/title/desc/answer）的样本，
    并将干净的数据写入新的 JSONL 文件。
    """

    required_keys = ["qid", "category", "title", "desc", "answer"]

    total = 0
    kept = 0
    removed = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            try:
                item = json.loads(line)
            except:
                removed += 1
                continue

            # 判断是否缺字段或字段为空
            missing = False
            for k in required_keys:
                if k not in item or item[k] in ["", None]:
                    missing = True
                    break

            if missing:
                removed += 1
                continue

            # 保留干净数据
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"总样本数: {total}")
    print(f"保留数量: {kept}")
    print(f"删除数量: {removed}")

    return kept, removed


if __name__ == '__main__':
    # count_missing_fields("../data/category_clean_valid.json")
    # clean_json_lines(
    #     "../data/category_clean_train.json",
    #     "../data/missing_clean_train.json"
    # )
    clean_json_lines(
        "../data/category_clean_valid.json",
        "../data/missing_clean_valid.json"
    )
