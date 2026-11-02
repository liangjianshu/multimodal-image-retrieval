# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
"""
ImageNet-21K 数据集构建脚本
按需求下载指定类别图片，划分常规/细粒度，生成标准JSON文件。

需求:
- 总计 2000 张: 正样本 200 + 负样本 1800
- 常规/细粒度 1:1 (各 1000 张)
- 正样本 5 类 x 40 张 (20常规 + 20细粒度)
- 负样本 10 类 x 180 张 (90常规 + 90细粒度)
- 细粒度: 目标面积占比 < 0.3
- 不足时互相补足
"""

import os
import json
import random
import shutil
import tarfile
import requests
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as TF
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ===================== 配置 =====================
ROOT_DIR = Path("/path/to/workspace/dataset/imagenet")
JSON_PATH = ROOT_DIR / "imagenet_json.json"
TEMP_DIR = ROOT_DIR / "_temp"
FINE_GRAIN_THRESHOLD = 0.3  # 目标面积占比 < 0.3 为细粒度

# 下载源（优先清华源）
DOWNLOAD_URLS = [
    "https://mirror.tuna.tsinghua.edu.cn/imagenet/data/CLS-LOC/train/{wnid}.tar",
    "https://image-net.org/data/winter21_whole/{wnid}.tar",
]

# 类别配置: (英文名, 中文名, WordNet ID, COCO ID)
POSITIVE_CLASSES = [
    # 通用类别
    ("chair",      "椅子",   "n03001627", 62),
    ("cup",        "杯子",   "n03147509", 47),
    ("bicycle",    "自行车", "n02834778",  2),
    # 业务类别
    ("skateboard", "滑板",   "n04225987", 41),
    ("oven",       "烤箱",   "n03862676", 79),
]

NEGATIVE_CLASSES = [
    # 难负样本（对应正样本5类）
    ("couch",      "沙发",     "n04256520", 63),
    ("bowl",       "碗",       "n02881193", 51),
    ("motorcycle", "摩托车",   "n03790512",  4),
    ("snowboard",  "滑雪板",   "n04251791", 36),
    ("microwave",  "微波炉",   "n03761084", 78),
    # 随机类别
    ("cat",        "猫",       "n02121620", 17),
    ("donut",      "甜甜圈",   "n07639069", 60),
    ("keyboard",   "键盘",     "n03614007", 76),
    ("orange",     "橙子",     "n07747607", 55),
    ("pizza",      "披萨",     "n07873807", 59),
]

# 数量要求
POS_PER_CLASS = 40       # 正样本每类 40 张
POS_NORMAL = 20          # 正样本每类常规 20 张
POS_FINE = 20            # 正样本每类细粒度 20 张
NEG_PER_CLASS = 180      # 负样本每类 180 张
NEG_NORMAL = 90          # 负样本每类常规 90 张
NEG_FINE = 90            # 负样本每类细粒度 90 张


# ===================== 下载 =====================
def download_wnid(wnid: str, save_dir: Path) -> bool:
    """下载并解压指定 wnid 的图片，优先清华源"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # 已有足够图片则跳过
    existing = list(save_dir.glob("*.JPEG")) + list(save_dir.glob("*.jpeg")) + list(save_dir.glob("*.jpg"))
    if len(existing) >= NEG_PER_CLASS:
        print(f"  [跳过] {wnid} 本地已有 {len(existing)} 张图片")
        return True

    tar_path = TEMP_DIR / f"{wnid}.tar"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 尝试各下载源
    if not tar_path.exists():
        for url_template in DOWNLOAD_URLS:
            url = url_template.format(wnid=wnid)
            try:
                print(f"  [下载] {wnid} <- {url}")
                resp = requests.get(url, stream=True, timeout=120)
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(tar_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=wnid, leave=False
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
                print(f"  [完成] {wnid} 下载成功")
                break
            except Exception as e:
                print(f"  [失败] {url} -> {str(e)[:80]}")
                tar_path.unlink(missing_ok=True)
                continue
        else:
            print(f"  [错误] {wnid} 所有下载源均失败")
            return False

    # 解压
    if tar_path.exists():
        try:
            print(f"  [解压] {wnid} -> {save_dir}")
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.lower().endswith((".jpeg", ".jpg")):
                        member.name = os.path.basename(member.name)
                        tar.extract(member, save_dir)
            return True
        except Exception as e:
            print(f"  [解压失败] {wnid}: {e}")
            return False
    return False


def download_all_classes():
    """并行下载所有类别（4线程加速）"""
    all_classes = POSITIVE_CLASSES + NEGATIVE_CLASSES
    print(f"=== 开始下载 {len(all_classes)} 个类别 ===\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for en, cn, wnid, coco_id in all_classes:
            save_dir = ROOT_DIR / en
            future = executor.submit(download_wnid, wnid, save_dir)
            futures[future] = (en, wnid)

        for future in as_completed(futures):
            en, wnid = futures[future]
            try:
                ok = future.result()
                if not ok:
                    print(f"  [警告] {en}({wnid}) 下载未成功，将使用已有图片")
            except Exception as e:
                print(f"  [异常] {en}({wnid}): {e}")

    print("\n=== 下载阶段完成 ===\n")


# ===================== 检测模型 =====================
DETECT_BATCH_SIZE = 8
DETECT_SCORE_THRESH = 0.3


def load_detector():
    """加载 Faster R-CNN (COCO 预训练)，全局只加载一次"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def _load_image_tensor(img_path: str):
    """读取图片并转为模型输入 tensor，返回 (tensor, w, h) 或 None"""
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        tensor = TF.to_tensor(img)
        return tensor, w, h
    except Exception:
        return None


def estimate_object_ratios_batch(img_paths: list, coco_id: int, model) -> list:
    """
    批量检测，返回每张图片中目标 COCO 类别最大检测框的面积占比。
    无检测结果返回 0.5（归为常规）。
    """
    device = next(model.parameters()).device
    results = [0.5] * len(img_paths)

    for start in range(0, len(img_paths), DETECT_BATCH_SIZE):
        batch_paths = img_paths[start:start + DETECT_BATCH_SIZE]
        batch_tensors = []
        batch_sizes = []  # (w, h)
        valid_indices = []  # 在 results 中的索引

        for i, p in enumerate(batch_paths):
            loaded = _load_image_tensor(str(p))
            if loaded is not None:
                tensor, w, h = loaded
                batch_tensors.append(tensor.to(device))
                batch_sizes.append((w, h))
                valid_indices.append(start + i)

        if not batch_tensors:
            continue

        with torch.no_grad():
            outputs = model(batch_tensors)

        for idx, out in enumerate(outputs):
            labels = out["labels"]
            scores = out["scores"]
            boxes = out["boxes"]

            # 筛选目标类别且置信度 > 阈值
            mask = (labels == coco_id) & (scores > DETECT_SCORE_THRESH)
            if mask.sum() == 0:
                results[valid_indices[idx]] = 0.5
                continue

            matched_boxes = boxes[mask]
            w, h = batch_sizes[idx]
            img_area = w * h

            # 计算每个框面积，取最大
            areas = (matched_boxes[:, 2] - matched_boxes[:, 0]) * (matched_boxes[:, 3] - matched_boxes[:, 1])
            max_area = areas.max().item()
            ratio = max_area / img_area
            results[valid_indices[idx]] = max(0.05, min(0.95, ratio))

    return results


# ===================== 图片筛选与划分 =====================
def get_all_images(cls_dir: Path) -> list:
    """获取目录下所有图片路径"""
    imgs = []
    for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
        imgs.extend(cls_dir.glob(ext))
    return imgs


def split_normal_fine(images: list, target_normal: int, target_fine: int, cls_name: str, coco_id: int, model):
    """
    将图片划分为常规/细粒度。
    使用 Faster R-CNN 按 COCO ID 检测目标面积占比。
    不足时从另一方补足，仍不足则重复采样。
    """
    normal_imgs = []
    fine_imgs = []

    # 批量推理
    ratios = estimate_object_ratios_batch(images, coco_id, model)

    for img_path, ratio in zip(images, ratios):
        if ratio >= FINE_GRAIN_THRESHOLD:
            normal_imgs.append(img_path)
        else:
            fine_imgs.append(img_path)

    random.shuffle(normal_imgs)
    random.shuffle(fine_imgs)

    n_normal = len(normal_imgs)
    n_fine = len(fine_imgs)

    # 互相补足
    if n_normal < target_normal and n_fine > target_fine:
        need = target_normal - n_normal
        normal_imgs.extend(fine_imgs[target_fine:target_fine + need])
    elif n_fine < target_fine and n_normal > target_normal:
        need = target_fine - n_fine
        fine_imgs.extend(normal_imgs[target_normal:target_normal + need])

    # 仍不足则重复采样
    if len(normal_imgs) < target_normal:
        pool = normal_imgs if normal_imgs else fine_imgs
        if pool:
            normal_imgs = random.choices(pool, k=target_normal)
    if len(fine_imgs) < target_fine:
        pool = fine_imgs if fine_imgs else normal_imgs
        if pool:
            fine_imgs = random.choices(pool, k=target_fine)

    normal_imgs = normal_imgs[:target_normal]
    fine_imgs = fine_imgs[:target_fine]

    print(f"  {cls_name}: 原始 {n_normal}常规+{n_fine}细粒度 -> 选取 {len(normal_imgs)}常规+{len(fine_imgs)}细粒度")
    return normal_imgs, fine_imgs


# ===================== 构建数据集 =====================
def process_classes(classes, target_per_class, target_normal, target_fine, start_id, label, model):
    """处理一组类别，返回图片列表、info字典、结束id"""
    images = []
    info = {}
    current_id = start_id

    print(f"=== 构建{label} ({len(classes)}类 x {target_per_class}张) ===")
    for en, cn, wnid, coco_id in classes:
        cls_dir = ROOT_DIR / en
        imgs = get_all_images(cls_dir)

        if not imgs:
            print(f"  [错误] {en} 无可用图片，跳过")
            continue

        # 采样
        if len(imgs) < target_per_class:
            print(f"  [警告] {en} 仅有 {len(imgs)} 张，需 {target_per_class} 张，将重复采样")
            sampled = random.choices(imgs, k=target_per_class)
        else:
            sampled = random.sample(imgs, target_per_class)

        normal_imgs, fine_imgs = split_normal_fine(sampled, target_normal, target_fine, en, coco_id, model)

        # info: [start_id, normal_end_id, total_end_id]
        info[en] = [current_id, current_id + len(normal_imgs), current_id + len(normal_imgs) + len(fine_imgs)]

        for img_path in normal_imgs:
            images.append({
                "id": current_id,
                "file_name": img_path.name,
                "file_path": str(img_path),
                "label": en,
                "label_cn": cn,
                "type": "normal"
            })
            current_id += 1

        for img_path in fine_imgs:
            images.append({
                "id": current_id,
                "file_name": img_path.name,
                "file_path": str(img_path),
                "label": en,
                "label_cn": cn,
                "type": "fine"
            })
            current_id += 1

    count = current_id - start_id
    print(f"  {label}总计: {count} 张\n")
    return images, info, current_id


def save_json(all_images, info):
    """生成 JSON 文件"""
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 完整数据集 (常规+细粒度)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"info": info, "images": all_images}, f, indent=4, ensure_ascii=False)
    print(f"  完整数据集 JSON -> {JSON_PATH}")

    # 仅常规数据集
    normal_images = [img for img in all_images if img["type"] == "normal"]
    normal_info = {}
    for label in info:
        label_normals = [img for img in normal_images if img["label"] == label]
        if label_normals:
            start = label_normals[0]["id"]
            end = label_normals[-1]["id"] + 1
            normal_info[label] = [start, end, end]


# ===================== 主函数 =====================
def main():
    print("=" * 60)
    print("  ImageNet-21K 数据集构建工具")
    print("=" * 60)
    print(f"  目标: 2000张 (正样本200 + 负样本1800)")
    print(f"  常规/细粒度: 1:1 (各1000张)")
    print(f"  保存路径: {ROOT_DIR}")
    print(f"  JSON路径: {JSON_PATH}")
    print("=" * 60 + "\n")

    # 1. 下载
    download_all_classes()

    # 2. 加载检测模型
    print("=== 加载 Faster R-CNN 检测模型 ===")
    detector = load_detector()
    print("  模型加载完成\n")

    # 3. 构建正样本
    pos_images, pos_info, next_id = process_classes(
        POSITIVE_CLASSES, POS_PER_CLASS, POS_NORMAL, POS_FINE, 0, "正样本", detector
    )

    # 4. 构建负样本
    neg_images, neg_info, final_id = process_classes(
        NEGATIVE_CLASSES, NEG_PER_CLASS, NEG_NORMAL, NEG_FINE, next_id, "负样本", detector
    )

    # 5. 合并并生成 JSON
    all_images = pos_images + neg_images
    all_info = {**pos_info, **neg_info}

    print("=== 生成 JSON 文件 ===")
    save_json(all_images, all_info)

    # 6. 清理临时文件
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("  临时文件已清理")

    # 7. 统计
    total = len(all_images)
    normal_count = sum(1 for img in all_images if img["type"] == "normal")
    fine_count = total - normal_count
    print(f"\n{'=' * 60}")
    print(f"  构建完成!")
    print(f"  总计: {total} 张 | 正样本: {len(pos_images)} | 负样本: {len(neg_images)}")
    print(f"  常规: {normal_count} 张 | 细粒度: {fine_count} 张")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    random.seed(42)
    main()
