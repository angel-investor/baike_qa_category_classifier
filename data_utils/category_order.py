import json

# 读取 JSON 文件
with open("../data/output_data/category_distribution_valid.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 按 value 降序排序
sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

# 打印或保存
print(json.dumps(sorted_data, ensure_ascii=False, indent=2))

# 如需写回文件：
with open("../data/output_data/category_distribution_valid_sort.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)