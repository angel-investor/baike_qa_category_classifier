import json
from collections import Counter

# 读取数据
with open("../data/pure_valid.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取所有类别
categories = [item['category'] for item in data]

# 统计类别数量
category_count = Counter(categories)

# 按降序排序
category_count_sorted = dict(sorted(category_count.items(), key=lambda x: x[1], reverse=True))

# 保存到文件
with open("../data/output_data/pure_category_count_valid.json", "w", encoding="utf-8") as f:
    json.dump(category_count_sorted, f, ensure_ascii=False, indent=2)

print("统计完成，已保存到 ../data/pure_category_count_valid.json")