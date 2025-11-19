import json

def check_category_diff(file_to_check, reference_file):
    """
    检查 file_to_check 中的 key 是否都存在于 reference_file 中。
    返回缺失的类别列表。

    Args:
        file_to_check (str): 待检查 JSON 文件路径
        reference_file (str): 参考 JSON 文件路径

    Returns:
        missing_keys (list): file_to_check 中缺失于 reference_file 的 key 列表
        """
    # 读取第一个 JSON 文件（需要检查的文件）
    with open(file_to_check, "r", encoding="utf-8") as f1:
        data1 = json.load(f1)

    # 读取第二个 JSON 文件（参考文件，包含完整的key集合）
    with open(reference_file, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)

    # data2 的 key 集合
    keys2 = set(data2.keys())

    # 遍历 data1 的 key，如果不在 data2 中就输出
    for key in data1.keys():
        # print(f"检查类别: {key}")

        if key not in keys2:
            print(f"缺失的类别: {key}")

if __name__ == '__main__':
    file_to_check = "../data/output_data/category_distribution_train.json"
    reference_file = "../data/output_data/category_distribution_valid.json"
    check_category_diff(file_to_check, reference_file)