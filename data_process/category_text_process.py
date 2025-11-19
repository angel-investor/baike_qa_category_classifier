import json

input_path = "../data/missing_clean_valid.json"
output_path = "../data/pure_valid.json"

def load_json(path):
    """
    兼容两种格式：
    1. 整个文件是一个 list
    2. 一行一个 json
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # 文件整体是一个 list
            return json.load(f)
        else:
            # 每行一个 JSON
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data


def process():
    data = load_json(input_path)
    print(f"读取到 {len(data)} 条样本")

    output = []
    for item in data:
        title = item.get("title", "").strip()
        desc = item.get("desc", "").strip()

        # 构造 text
        text = title + " " + desc

        output.append({
            "category": item.get("category", ""),
            "text": text
        })

    # 保存为 JSON 数组
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"处理完成，共写入 {len(output)} 条到 {output_path}")


if __name__ == "__main__":
    process()
