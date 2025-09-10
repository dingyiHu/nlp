import json

# 读取 JSON 文件
with open("/root/autodl-tmp/shiyan/dataset/result/bio_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 用来存放所有 BIO 标签
all_labels = []

# 遍历每条数据
for item in data:
    all_labels.extend(item["label"])  # 把每条的 label 列表加入总列表

# 如果只想要唯一标签
unique_labels = list(set(all_labels))

print("所有标签:", all_labels)
print("唯一标签:", unique_labels)
