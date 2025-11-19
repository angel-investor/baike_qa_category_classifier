import json

category_file = "../data/output_data/category_distribution_train_sort.json"
src_path = "../data/baike_qa_valid.json"
dst_path = "../data/category_clean_valid.json"

# 1. 读取类别数量
with open(category_file, "r", encoding="utf-8") as f:
    category_cnt = json.load(f)

# 2. 找出类别数量 < 500 的类别
remove_labels = {label for label, cnt in category_cnt.items() if cnt < 500}
print("训练集中低于 500 的类别数:", len(remove_labels))

# 3. 过滤原始数据
keep_cnt = 0
remove_cnt = 0

with open(src_path, "r", encoding="utf-8") as fin, \
     open(dst_path, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)
        label = item["category"]  # 如果字段名不同，改这里

        if label in remove_labels:
            remove_cnt += 1
            continue

        fout.write(line)
        keep_cnt += 1

print("保留数据:", keep_cnt)
print("删除数据:", remove_cnt)
print("新文件已保存:", dst_path)