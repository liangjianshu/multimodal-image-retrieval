# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
import os
import numpy as np
import torch
import json
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 常规dataset=1,常规+细粒度dataset=2
dataset = 2
# 参考图数量 1/2/5/10
image_number = 0
# 是否和prompt文本特征融合 True/False
with_prompt = True
model_type = "taiyi_clip"
dataset_type = ["常规", "常规+细粒度"]
test_model = None
text_tokenizer_pt = None
processor_pt = None
get_image_features_pt = None
get_text_features_pt = None

path = "./imagenet_json.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.keys())
print(data["info"])
print(data["images"][0])

if model_type == "clip":
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

    print(f"正在加载【Clip】模型...")
    test_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    text_tokenizer_pt = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    processor_pt = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    get_text_features_pt = test_model.get_text_features
    get_image_features_pt = test_model.get_image_features

elif model_type == "chinese_clip":
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name

    print(f"正在加载【Chinese Clip】模型...")
    test_model, processor_pt = load_from_name("ViT-B-16", device=device, download_root='./')
    text_tokenizer_pt = clip.tokenize
    get_text_features_pt = test_model.encode_text
    get_image_features_pt = test_model.encode_image

elif model_type == "taiyi_clip":
    from transformers import BertForSequenceClassification, BertTokenizer
    from transformers import CLIPProcessor, CLIPModel

    print(f"正在加载【Taiyi Clip】模型...")
    test_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    text_tokenizer_pt = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
    processor_pt = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    get_text_features_pt = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval().to(device)
    get_image_features_pt = test_model.get_image_features

elif model_type == "siglip_v2":
    from transformers import AutoModel, AutoProcessor

    print(f"正在加载【Siglip v2】模型...")
    ckpt = "google/siglip2-base-patch16-224"
    test_model = AutoModel.from_pretrained(ckpt, device_map="auto",torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).eval()
    processor_pt = AutoProcessor.from_pretrained(ckpt)
    get_text_features_pt = test_model.get_text_features
    get_image_features_pt = test_model.get_image_features

else:
    raise ValueError(f"不支持的模型类型：{model_type}！可选值：chinese_clip/taiyi_clip/siglip_v2")

def text_tokenizer(prompt):
    match model_type:
        case "clip" | "taiyi_clip":
            tokens = text_tokenizer_pt(
                prompt,
                return_tensors="pt",
                padding=True
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            return tokens
        case "siglip_v2":
            tokens = processor_pt(
                text=prompt,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)
            return tokens
        case "chinese_clip":
            if isinstance(prompt, str):
                prompt = [prompt]
            tokens = text_tokenizer_pt(prompt).to(device)
            return tokens
        case _:
            raise ValueError("Unknown model type")

def processor(images):
    match model_type:
        case "clip" | "taiyi_clip":
            inputs = processor_pt(images=images, return_tensors="pt")
            return inputs["pixel_values"].to(device)

        case "siglip_v2":
            img = processor_pt(
                images=images,
                return_tensors="pt"
            ).to(device)
            return img

        case "chinese_clip":
            img = processor_pt(images).unsqueeze(0).to(device)
            return img
        case _:
            raise ValueError("Unknown model type")

def get_text_features(text_inputs):
    match model_type:
        case "clip":
            return get_text_features_pt(**text_inputs)

        case "taiyi_clip":
            return get_text_features_pt(**text_inputs).logits

        case "siglip_v2":
            return get_text_features_pt(**text_inputs).pooler_output

        case "chinese_clip":
            return get_text_features_pt(text_inputs)
        case _:
            raise ValueError("Unknown model type")

def get_image_features(image_inputs):
    match model_type:
        case "clip" | "taiyi_clip":
            return get_image_features_pt(pixel_values=image_inputs)

        case "siglip_v2":
            return get_image_features_pt(pixel_values=image_inputs["pixel_values"]).pooler_output

        case "chinese_clip":
            return get_image_features_pt(image_inputs)
        case _:
            raise ValueError("Unknown model type")

def eval(pos_res, neg_res, threshold):
    p, n = np.array(pos_res), np.array(neg_res)
    tp = sum(p >= threshold)
    fp = sum(n >= threshold)
    fn = sum(p < threshold)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0

    return f1_score, precision, recall

def clip_by_threshold(positive_cate,image_number,prompt,dataset):
    pos_res, neg_res = [], []
    pos_fine_res = []

    features_match = None
    # 文本
    if prompt[1]:
      text = text_tokenizer(f"{prompt[0]}")
      with torch.no_grad():
        text_features = get_text_features(text)
        text_features = text_features / torch.linalg.norm(text_features, dim=1, keepdim=True)
        features_match = text_features if features_match is None else features_match + text_features

    # 图像
    image_data = data["images"]
    ref_image_range = []
    if image_number != 0:
        ref_image_range = range(data["info"][positive_cate][0],data["info"][positive_cate][0]+image_number)
        for i in ref_image_range:
            image = processor(images=Image.open(image_data[i]["file_path"]))
            with torch.no_grad():
                image_features = get_image_features(image)
                image_features = image_features / torch.linalg.norm(image_features, dim=1, keepdim=True)
                features_match = image_features if features_match is None else features_match + image_features

    # 归一化
    features_match = features_match / torch.linalg.norm(features_match, dim=1, keepdim=True).to(device)

    pos_list, neg_list = [], []
    for cate_name in data["info"]:
        if cate_name == positive_cate:
            index = data["info"][cate_name]
            end_idx = index[1] if dataset == 1 else index[2]
            pos_list.append(range(index[0], end_idx))
        else:
            index = data["info"][cate_name]
            end_idx = index[1] if dataset == 1 else index[2]
            neg_list.append(range(index[0], end_idx))

    pos_list = [x for r in pos_list for x in r if x not in ref_image_range]
    neg_list = [x for r in neg_list for x in r]

    if not set(pos_list) & set(neg_list):
        print(f"pos_list和neg_list互斥，分别包含的图片数为{len(pos_list)},{len(neg_list)}")
    else:
        print("pos_list和neg_list有交集")
        print(set(pos_list) & set(neg_list))

    # 遍历正样本
    for m in pos_list:
        image = processor(images=Image.open(image_data[m]["file_path"]))
        with torch.no_grad():
            image_features = get_image_features(image)
            image_features = image_features / torch.linalg.norm(image_features, dim=1, keepdim=True)
            if model_type == "siglip_v2":
                logits_per_image = test_model.logit_scale.exp() * image_features @ features_match.t() + test_model.logit_bias
            else:
                logits_per_image = (test_model.logit_scale.exp()) * image_features @ features_match.t()
            pos_res.extend(logits_per_image.cpu().numpy()[0])
            if image_data[m]["type"] == "fine":
                pos_fine_res.extend(logits_per_image.cpu().numpy()[0])

    # 遍历负样本
    for m in neg_list:
        image = processor(images=Image.open(image_data[m]["file_path"]))
        with torch.no_grad():
            image_features = get_image_features(image)
            image_features = image_features / torch.linalg.norm(image_features, dim=1, keepdim=True)
            if model_type == "siglip_v2":
                logits_per_image = test_model.logit_scale.exp() * image_features @ features_match.t() + test_model.logit_bias
            else:
                logits_per_image = (test_model.logit_scale.exp()) * image_features @ features_match.t()
            neg_res.extend(logits_per_image.cpu().numpy()[0])

    print(f"正样本数量：{len(pos_res)}, 负样本数量：{len(neg_res)}")
    return pos_res, neg_res, pos_fine_res

def main(postive_cate,image_number,prompt,dataset):
    pos_res, neg_res, pos_fine_res = clip_by_threshold(postive_cate,image_number,prompt,dataset)
    cate_name = postive_cate

    # Top-K precision
    K=[1,5,10,20]
    all_scores = []
    for score in pos_res:
        all_scores.append((score, 1))
    for score in neg_res:
        all_scores.append((score, 0))
    sorted_items = sorted(all_scores, key=lambda x: x[0], reverse=True)
    k_precision = []
    for k in K:
        if k > len(sorted_items):
            k_precision.append(0.0)
            continue
        top_k = sorted_items[:k]
        num_pos_in_top_k = sum([label for score, label in top_k])
        k_precision.append(num_pos_in_top_k / k)
    print(f"{cate_name} Top-{K} precision:", k_precision)

    # 阈值遍历
    thresholds = np.linspace(min(min(pos_res), min(neg_res)), max(max(pos_res), max(neg_res)), 50)
    best_threshold, best_f1_score, best_precision, best_recall = 0., 0., 0., 0.
    best_fine_recall = 0.0
    f1_scores = []

    for threshold in thresholds:
        f1_score, precision, recall = eval(pos_res, neg_res, threshold)
        f1_scores.append(f1_score)
        current_fine_recall = sum(1 for s in pos_fine_res if float(s) >= float(threshold)) / len(pos_fine_res) if pos_fine_res else 0
        if f1_score > best_f1_score:
            best_threshold = threshold
            best_f1_score = f1_score
            best_precision = precision
            best_recall = recall
            best_fine_recall = current_fine_recall

    print(f"{cate_name}_best_threshold: {best_threshold:.3f}")
    print(f"{cate_name}_best_precision: {best_precision:.3f}")
    print(f"{cate_name}_best_recall: {best_recall:.3f}")
    print(f"{cate_name}_best_f1_score: {best_f1_score:.3f}")
    print(f"{cate_name}_best_fine_recall: {best_fine_recall:.3f}")

    return k_precision, [best_threshold, best_precision, best_recall, best_f1_score, best_fine_recall]

if __name__ == "__main__":
    pos_table = ["chair", 'cup', 'bicycle', 'skateboard', 'oven']
    pos_table_cn = ["椅子", '杯子', '自行车', '滑板', '烤箱']

    f1 = [[0] * 5 for _ in range(5)]
    k_pre = [[0] * 4 for _ in range(5)]

    for i in range(0, 5):
        print(f"\n===== 处理类别：{pos_table[i]} =====")
        if model_type == "clip" or model_type == "siglip_v2":
            prompt1 = pos_table[i]
            prompt2 = f"A photo of a {pos_table[i]}"
        else:
            prompt1 = pos_table_cn[i]
            prompt2 = "这是一张"+pos_table_cn[i]+"的照片"
        k_pre[i], f1[i] = main(postive_cate=pos_table[i],image_number=image_number,prompt=[prompt2,with_prompt], dataset=dataset)

    # 计算平均值
    labels_k_pre = ['K=1','K=5','K=10','K=20']
    k_pre.append(np.mean(k_pre, axis=0).tolist())
    k_pre.insert(0, labels_k_pre)

    labels_f1 = ["Best Threshold","Precision","Recall","F1-Score","fine_recall"]
    f1.append(np.mean(f1, axis=0).tolist())
    f1.insert(0, labels_f1)

    # 打印结果
    print(f'\n参考图n={image_number}/{dataset_type[dataset - 1]}/{model_type}/是否加入文本检索:{with_prompt}')

    print('Top-K Precision')
    for col in zip(*k_pre):
        for x in col:
            if isinstance(x, (int, float, np.floating)):
                print(f"{float(x):<10.3f}",end=' ')
            else:
                print(f"{str(x):<15}",end=' ')
        print()
    print('F1 Scores')
    for col in zip(*f1):
        for x in col:
            if isinstance(x, (int, float, np.floating)):
                print(f"{float(x):<10.3f}",end=' ')
            else:
                print(f"{str(x):<15}",end=' ')
        print()
