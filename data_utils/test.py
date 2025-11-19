import json

json_path = "../data/output_data/category_distribution_train.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 降序排列数量
sorted_counts = sorted(data.values(), reverse=True)

print(sorted_counts)
print("总数量：", len(sorted_counts))
