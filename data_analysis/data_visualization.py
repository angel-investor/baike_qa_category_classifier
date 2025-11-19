import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================
# 1. 读取 JSON 文件
# ==========================
json_path = "../data/output_data/category_distribution_train.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 按数量降序排序
sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)


# ==========================
# 2. 可视化 TOP50 柱状图 + 保存
# ==========================
top_k = 50
top_items = sorted_items[-top_k:]

categories = [item[0] for item in top_items]
counts = [item[1] for item in top_items]

plt.figure(figsize=(16, 10))
plt.barh(categories[::-1], counts[::-1])  # 最大数量显示在最上面
plt.title(f"Last {top_k} Categories Distribution")
plt.xlabel("Count")

# === 新增：为每一条柱子标注数量 ===
for idx, value in enumerate(counts[::-1]):
    plt.text(value + max(counts) * 0.01,
             idx,
             str(value),
             va='center', fontsize=9)

plt.tight_layout()

plt.savefig("../figure/last50.png", dpi=300, bbox_inches="tight")
print("已保存：top50.png")
plt.close()


# ==========================
# 3. 生成词云图 + 保存
# ==========================
wc = WordCloud(
    width=2000,
    height=1200,
    background_color="white",
    font_path="simhei.ttf"
)

wc.generate_from_frequencies(data)

plt.figure(figsize=(14, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Category WordCloud")

plt.savefig("../figure/wordcloud.png", dpi=300, bbox_inches="tight")
print("已保存：wordcloud.png")

plt.close()
