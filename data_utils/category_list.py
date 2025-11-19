import json

json_path = "../data/output_data/final_category_count_train_sort.json"
txt_output = "../data/category_list.txt"

# 读取 JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取所有 key（类别名称）
categories = list(data.keys())

# 写入 txt（一行一个类别）
with open(txt_output, "w", encoding="utf-8") as f:
    for cat in categories:
        f.write(cat + "\n")

print(f"完成！共写入 {len(categories)} 个类别到：{txt_output}")
