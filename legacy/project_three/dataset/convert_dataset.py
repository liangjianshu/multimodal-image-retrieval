# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
import json
import re

with open("finetune_dataset.json", "r") as f:
    data = json.load(f)

converted = []
for item in data:
    images = []
    new_convs = []
    for turn in item["conversations"]:
        role = "human" if turn["from"] == "user" else turn["from"]
        text = turn["value"]
        # Extract image paths from <|vision_start|>...<|vision_end|>
        found = re.findall(r"<\|vision_start\|>(.*?)<\|vision_end\|>", text)
        for img_path in found:
            img_path = img_path.lstrip("./")
            images.append(img_path)
            text = text.replace(f"<|vision_start|>{found[images.index(img_path) - len(images)]}<|vision_end|>", "", 1)
        # Replace all vision tags with <image>
        text = re.sub(r"<\|vision_start\|>.*?<\|vision_end\|>", "<image>", turn["value"])
        new_convs.append({"from": role, "value": text})

    new_item = {"id": item["id"], "conversations": new_convs}
    if images:
        new_item["image"] = images
    converted.append(new_item)

with open("finetune_dataset_converted.json", "w") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"Converted {len(converted)} samples")
print("Sample:", json.dumps(converted[0], indent=2, ensure_ascii=False))
