import json

# 读取原始数据
input_path = "../data/pure_valid.json"
output_path = "../data/pure_valid_processed.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

processed_data = []
for item in data:
    cat = item.get("category", "")

    # 删除健康-妇产科-产科
    if cat.startswith("健康-妇产科-产科"):
        continue

    # 合并肿瘤科子类和人体常识到健康
    if cat in ["健康-肿瘤科",
               "健康-肿瘤科-宫颈癌",
               "健康-肿瘤科-膀胱癌",
               "健康-肿瘤科-直肠癌",
               "健康-人体常识"]:
        item["category"] = "健康"

    processed_data.append(item)

# 保存处理后的数据
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print(f"处理完成，原数据条数: {len(data)}，处理后条数: {len(processed_data)}")
