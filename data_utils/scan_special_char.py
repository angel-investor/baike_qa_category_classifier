import unicodedata

filename = "../data/baike_qa_valid.json"
output_file = "../data/output_data/special_chars_result_valid.txt"

# 控制字符范围：0x00–0x1F 以及 0x7F
def is_control_char(ch):
    return (0 <= ord(ch) <= 31) or (ord(ch) == 127)

special_chars = {}

# 读取文件
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

# 扫描特殊字符
for ch in text:
    if is_control_char(ch) or ch in ['\u3000']:
        special_chars[ch] = special_chars.get(ch, 0) + 1

# 打印到终端
print("发现的特殊字符：")

# 也保存到文件
with open(output_file, "w", encoding="utf-8") as out:
    out.write("发现的特殊字符：\n\n")

    for ch, count in special_chars.items():
        name = unicodedata.name(ch, "UNKNOWN")
        line = f"字符: {repr(ch)} | Unicode: U+{ord(ch):04X} | 名称: {name} | 次数: {count}\n"

        print(line, end="")
        out.write(line)

print(f"\n结果已保存到: {output_file}")