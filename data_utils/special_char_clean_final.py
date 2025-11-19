import json

# ==========================
# 去除特殊符号函数
# ==========================
def clean_text(s):
    return (
        s.replace('\r', '')     # 去掉回车符
        .replace('\n', '')       # 去掉换行符
         .replace('', '')        # 去掉杂乱全角空格
         .replace('\u3000', '')   # 去掉全角空格
         .replace('\x7f', '')     # 控制字符
         .strip()
    )

# ==========================
# 主处理逻辑
# ==========================
in_path = "../data/final_test.json"

with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("原始数据量：", len(data))

new_data = []

for item in data:
    text = item.get("text", "")

    # 清洗文本
    text_clean = clean_text(text)

    # 跳过长度>500的
    if len(text_clean) > 500:
        continue

    # 写回
    item["text"] = text_clean
    new_data.append(item)

print("处理后数据量：", len(new_data))

# ==========================
# 原地覆盖写回
# ==========================
with open(in_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("处理完成，文件已原地更新：", in_path)
