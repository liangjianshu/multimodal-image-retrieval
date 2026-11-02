# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
"""
对 finetune_dataset.json 的 assistant 回复做文本多样化，
不改动图片路径，只替换文本模板，生成新文件。
"""
import json, re, random, copy

SRC = "finetune_dataset.json"
DST = "finetune_dataset_aug.json"

# 多样化模板，{cat} 占位
TEMPLATES = [
    "This is a {cat}.",
    "A photo of a {cat}.",
    "A {cat} in the image.",
    "The image shows a {cat}.",
    "This image contains a {cat}.",
    "A {cat} can be seen in this photo.",
    "There is a {cat} in the picture.",
    "The picture features a {cat}.",
]

# 匹配现有的两种句式，提取类别
PATTERN = re.compile(
    r"(?:This is a |This image shows a? ?|A photo of a )(\w[\w\s]*)\.?$",
    re.IGNORECASE,
)

with open(SRC, "r") as f:
    data = json.load(f)

changed = 0
for sample in data:
    for turn in sample["conversations"]:
        if turn["from"] in ("assistant", "gpt"):
            m = PATTERN.match(turn["value"].strip())
            if m:
                cat = m.group(1).strip()
                turn["value"] = random.choice(TEMPLATES).format(cat=cat)
                changed += 1

with open(DST, "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"共替换 {changed}/{len(data)} 条，已保存到 {DST}")
