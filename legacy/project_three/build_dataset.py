# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pycocotools.coco import COCO
import requests

# ===================== 【核心配置】 =====================
# 已下载图片的根目录
IMAGENET_SRC = Path("/path/to/workspace/dataset/imagenet")

# 正样本（你要的类别）
POS_CLASSES = {
    "chair": 200,
    "cup": 200,
    "bicycle": 200,
    "skateboard": 200,
    "oven": 200,
}
# 难负样本（相似类）
HARD_NEG_CLASSES = {
    "couch": 100,
    "bowl": 100,
    "motorcycle": 100,
    "snowboard": 100,
    "microwave": 100,
}
# 随机负样本（不相关）
RANDOM_NEG_CLASSES = {
    "cat": 100,
    "donut": 100,
    "orange": 100,
    "keyboard": 100,
    "pizza": 100,
}

TOTAL_IMAGENET21K = 2000
TOTAL_COCO = 2000
OUT_DIR = Path("./dataset")
IMG_DIR = OUT_DIR / "imagenet21k"
COCO_IMG_DIR = OUT_DIR / "coco"
IMG_DIR.mkdir(parents=True, exist_ok=True)
COCO_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 工具函数 =====================
def get_all_images(cls_dir: Path) -> list:
    """获取目录下所有图片路径"""
    imgs = []
    for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
        imgs.extend(cls_dir.glob(ext))
    return imgs


def download_coco_image(url, save_path):
    """下载并验证图片"""
    if os.path.exists(save_path):
        return True
    try:
        r = requests.get(url, timeout=10, stream=True)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            Image.open(save_path).verify()
            return True
    except Exception:
        return False
    return False


def build_sample(img_rel_path, desc, sample_id):
    """Qwen-VL 对话格式"""
    return {
        "id": sample_id,
        "conversations": [
            {
                "from": "user",
                "value": f"Describe this image: <|vision_start|>{img_rel_path}<|vision_end|>"
            },
            {
                "from": "assistant",
                "value": desc
            }
        ]
    }


# ===================== 1) 生成 ImageNet-21K 2000样本 =====================
def build_imagenet21k():
    samples = []
    idx = 0

    # 汇总所有类别: (类名, 需要数量, 角色)
    all_classes = {}
    for name, need in POS_CLASSES.items():
        all_classes[name] = {"need": need, "role": "pos", "collected": []}
    for name, need in HARD_NEG_CLASSES.items():
        all_classes[name] = {"need": need, "role": "hard_neg", "collected": []}
    for name, need in RANDOM_NEG_CLASSES.items():
        all_classes[name] = {"need": need, "role": "rand_neg", "collected": []}

    total_need = sum(v["need"] for v in all_classes.values())
    print(f"从本地 {IMAGENET_SRC} 读取，目标 {len(all_classes)} 个类别，共需 {total_need} 张图片\n")

    for cls_name, info in all_classes.items():
        need = info["need"]
        cls_dir = IMAGENET_SRC / cls_name

        if not cls_dir.exists():
            print(f"  [警告] {cls_name} 目录不存在: {cls_dir}，跳过")
            continue

        all_imgs = get_all_images(cls_dir)
        if not all_imgs:
            print(f"  [警告] {cls_name} 无可用图片，跳过")
            continue

        # 随机采样
        random.shuffle(all_imgs)
        sampled = all_imgs[:need] if len(all_imgs) >= need else random.choices(all_imgs, k=need)

        print(f"[{cls_name}] 可用 {len(all_imgs)} 张，采样 {len(sampled)} 张")

        # 转存为 jpg
        for img_path in tqdm(sampled, desc=f"  处理 {cls_name}", leave=False):
            try:
                img = Image.open(img_path).convert("RGB")
                save_name = f"{cls_name}_{idx}.jpg"
                save_path = IMG_DIR / save_name
                img.save(save_path, "JPEG")
                rel_path = f"./images/imagenet21k/{save_name}"
                info["collected"].append((rel_path, idx))
                idx += 1
            except Exception:
                continue

    # 生成样本记录
    pos_names = list(POS_CLASSES.keys())
    for cls_name, info in all_classes.items():
        for rel_path, sample_idx in info["collected"]:
            if info["role"] == "pos":
                desc = f"This image shows {cls_name}."
                sid = f"imagenet21k_pos_{sample_idx}"
            else:
                wrong_name = random.choice(pos_names)
                desc = f"This image shows a {wrong_name}."
                prefix = "hardneg" if info["role"] == "hard_neg" else "randneg"
                sid = f"imagenet21k_{prefix}_{sample_idx}"
            samples.append(build_sample(rel_path, desc, sid))

    random.shuffle(samples)
    print(f"\n各类别收集情况:")
    for cls_name, info in all_classes.items():
        print(f"  {cls_name}: {len(info['collected'])}/{info['need']}")
    return samples[:TOTAL_IMAGENET21K]


# ===================== 2) 生成 COCO 2000样本（不变） =====================
def build_coco():
    samples = []
    coco_anno = "annotations/instances_train2017.json"
    if not os.path.exists(coco_anno):
        os.system("wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O ann.zip")
        os.system("unzip -o -q ann.zip && rm ann.zip")
    coco = COCO(coco_anno)

    coco_pos = ["chair", "cup", "bicycle", "skateboard", "oven"]
    coco_hard_neg = ["couch", "bowl", "motorcycle", "snowboard", "microwave"]
    coco_rand_neg = ["cat", "donut", "keyboard", "orange", "pizza"]

    # 正样本1000
    for cat in coco_pos:
        cat_id = coco.getCatIds(catNms=[cat])[0]
        img_ids = coco.getImgIds(catIds=[cat_id])
        random.shuffle(img_ids)
        for img_id in tqdm(img_ids[:200], desc=f"  COCO正样本 {cat}", leave=False):
            img = coco.loadImgs(img_id)[0]
            path = f"{COCO_IMG_DIR}/coco_{img_id}.jpg"
            rel_path = f"./images/coco/coco_{img_id}.jpg"
            download_coco_image(img["coco_url"], path)
            samples.append(build_sample(rel_path, f"This is a {cat}.", f"coco_pos_{img_id}"))

    # 难负500 + 随机负500
    for cats in [coco_hard_neg, coco_rand_neg]:
        for cat in cats:
            cat_id = coco.getCatIds(catNms=[cat])[0]
            img_ids = coco.getImgIds(catIds=[cat_id])
            random.shuffle(img_ids)
            for img_id in tqdm(img_ids[:100], desc=f"  COCO负样本 {cat}", leave=False):
                img = coco.loadImgs(img_id)[0]
                path = f"{COCO_IMG_DIR}/coco_neg_{img_id}.jpg"
                rel_path = f"./images/coco/coco_neg_{img_id}.jpg"
                download_coco_image(img["coco_url"], path)
                wrong = random.choice(coco_pos)
                samples.append(build_sample(rel_path, f"This is a {wrong}.", f"coco_neg_{img_id}"))

    return samples[:TOTAL_COCO]


# ===================== 主流程 =====================
if __name__ == "__main__":
    print("=== 生成 ImageNet-21K 2000样本 ===")
    data_21k = build_imagenet21k()
    print(f"21K 完成：{len(data_21k)}")

    print("\n=== 生成 COCO 2000样本 ===")
    data_coco = build_coco()
    print(f"COCO 完成：{len(data_coco)}")

    # 合并+打乱
    all_data = data_21k + data_coco
    random.shuffle(all_data)

    # 保存JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "finetune_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n全部完成！总计：{len(all_data)}")
    print(f"JSON：{OUT_DIR / 'finetune_dataset.json'}")
