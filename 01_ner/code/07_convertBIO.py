import json

def convert_to_bio_labels(item):
    text = item['text']
    words = item['words']
    n = len(words)
    bio_labels = ['O'] * n

    for entity_type, entities in item['label'].items():
        for ent_name, spans in entities.items():
            for start, end in spans:
                if start >= n or end >= n:
                    continue
                bio_labels[start] = f'B-{entity_type}'
                for i in range(start + 1, end + 1):
                    bio_labels[i] = f'I-{entity_type}'

    new_item = {
        'text': text,
        'words': words,
        'label': bio_labels
    }
    if 'semantic_matrix' in item:
        new_item['semantic_matrix'] = item['semantic_matrix']

    return new_item

# -------------------------------
# 读取 JSON 文件并转换
# -------------------------------
input_file = '/root/autodl-tmp/shiyan/dataset/cluener_public/whole.json'       # 你的原始文件
output_file = '/root/autodl-tmp/shiyan/dataset/cluener_public/whole_bio.json'  # 输出文件

converted_list = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)  # 逐行解析
        converted_list.append(convert_to_bio_labels(item))

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_list, f, ensure_ascii=False, indent=2)

print(f'转换完成，结果已保存到 {output_file}')




