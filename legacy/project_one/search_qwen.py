# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch, json, torch, re, random, time, ast, os
from peft import PeftModel
import numpy as np
from PIL import Image

# 选择用于评估的特征key
# 测试类别准确率用 "category_feature"
# 测试描述细粒度用 "Qwen_features1" / "Qwen_features2" / "Qwen_features3"
labels = "category_feature"
# 常规dataset=1,常规+细粒度dataset=2
dataset_type = ["常规", "常规+细粒度"]
dataset = 1
# 参考图数量 1/2/5/10
image_number = 0
# 是否和prompt文本特征融合 True/False
with_prompt = True
# 是否使用lora微调 True/False
if_lora = True

model_path = "/path/to/workspace/multimodel_project/model-Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="cuda",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)

# 选择是否加入最优lora微调
if if_lora:
  print("加入模型微调")
  lorapath = '/path/to/workspace/multimodel_project/project_one/output_lora_llm'
  model = PeftModel.from_pretrained(
      model,
      lorapath   # 含 adapter_config.json 的目录
  ).merge_and_unload()

# 读取测试集json文件
path1 = './imagenet_json.json'
with open(path1, "r", encoding="utf-8") as f:
  data = json.load(f)

# 类别识别prompt（开放式，输出主要物体类别名）
category_prompt = '''
<image>
根据图片内容，输出图中包含的物品类别列表。
严格按照以下列表格式输出：
["类别1","类别2","..."]
'''

# 描述prompt（输出图片描述，用于细粒度检索）
prompt1 = '''
<image>
请用一句简短的中文描述图片中的主要内容，包括物体类型和场景。
例如：”一把放在客厅里的红色木质椅子”
只输出描述内容，不要输出其他信息。
'''
prompt2='''
<image>
请用中文详细描述图片中的内容，包括：物体类型、颜色、材质、形状、场景环境等。
例如：”一辆黑色的山地自行车停在公园的草地上，车身有红色条纹，轮胎较粗”
只输出描述内容，不要输出其他信息。
'''
prompt3 = '''
<image>
请用中文描述图片中最显著的物体，重点描述其类别、外观特征和所处环境。
例如：”图中是一个白色的陶瓷咖啡杯，放在木质桌面上，旁边有一本书”
只输出描述内容，不要输出其他信息。
'''
def get_messages(image,prompt):
  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": image,
              },
              {
                  "type": "text",
                  "text": prompt,
              },
              ],
          }
      ]
  return messages

# 推理函数
def get_response(messages):
  text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
      )
  image_inputs, video_inputs = process_vision_info(messages)
  inputs = processor(
      text=[text],
      images=image_inputs,
      videos=video_inputs,
      padding=True,
      return_tensors="pt",
  )
  inputs = inputs.to("cuda")

  # Inference: Generation of the output
  with torch.no_grad():
    with torch.autocast("cuda"):
      generated_ids = model.generate(**inputs, max_new_tokens=256)
  generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
  ]
  output_text = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
      # generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, do_sample=True, temperature=0.2,
  )
  return output_text[0]

# 分别用prompt1和prompt2为每张图片提取细粒度标签
import time
start = time.time()
if if_lora:
  save_path = './Qwenlora_Labels.json'
else:
  save_path = './Qwen_Labels.json'
if os.path.exists(save_path):
  print("已存在标签文件，直接读取跳过推理")
  with open(save_path, "r", encoding="utf-8") as f:
    data = json.load(f)
else:
  for index in range(0,len(data["images"])):
    # 类别识别（开放式，短类别名）—— 清理掉 [] " 等非法字符，只保留纯类别名
    raw_category = get_response(get_messages(data["images"][index]["file_path"],category_prompt)).strip()
    data["images"][index]["Qwen_category"] = re.sub(r'[\[\]\"\'\\]', '', raw_category).strip()
    # 描述生成（三种粒度）
    data["images"][index]["Qwen_prompt1_labels"] = get_response(get_messages(data["images"][index]["file_path"],prompt1)).strip()
    data["images"][index]["Qwen_prompt2_labels"] = get_response(get_messages(data["images"][index]["file_path"],prompt2)).strip()
    data["images"][index]["Qwen_prompt3_labels"] = get_response(get_messages(data["images"][index]["file_path"],prompt3)).strip()
    if index % 100 == 0:
      end = time.time()
      print(index)
      print(f"耗时: {end - start:.4f} 秒")
  # 将data写入json文件
  with open(save_path, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
# 加载Taiyi中文text encoder
text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
text_encoder = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval().to("cuda")
# 加载CLIP的image encoder
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 定义Taiyi-CLIP的标签特征提取函数
def get_labels_features(labels):
  if not labels:
    return torch.zeros([1,768]).to("cuda")
  if isinstance(labels, str):
    labels = [labels]
  text = text_tokenizer(labels, return_tensors='pt', padding=True)['input_ids'].to("cuda")
  with torch.no_grad():
    text_features = text_encoder(text).logits
    text_features = text_features / torch.linalg.norm(text_features, dim=1, keepdim=True)
  return text_features

print(get_labels_features(data["images"][1800]["Qwen_prompt1_labels"]).shape)
print(get_labels_features(data["images"][1800]["Qwen_prompt2_labels"]).shape)
print(get_labels_features(data["images"][1800]["Qwen_prompt3_labels"]).shape)
print(data["images"][1800].keys())

# 计算特征并存储
if if_lora:
  save_path = './Qwenlora_Labels_features.json'
else:
  save_path = './Qwen_Labels_features.json'
if os.path.exists(save_path):
  print("已存在标签文件，直接读取跳过推理")
  with open(save_path, "r", encoding="utf-8") as f:
    data = json.load(f)
else:
  for i in range(len(data["images"])):
    sample = data["images"][i]
    # 类别特征（短类别名向量，用于筛选）
    data["images"][i]["category_feature"] = get_labels_features(sample["Qwen_category"]).tolist()
    # 描述特征（描述向量，用于精排）
    data["images"][i]["Qwen_features1"] = get_labels_features(sample["Qwen_prompt1_labels"]).tolist()
    data["images"][i]["Qwen_features2"] = get_labels_features(sample["Qwen_prompt2_labels"]).tolist()
    data["images"][i]["Qwen_features3"] = get_labels_features(sample["Qwen_prompt3_labels"]).tolist()
    if i % 500 == 0:
      print(i)
    # .tolist()
  # 将data写入json文件
  with open(save_path, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)

def clip_by_threshold(positive_cate,image_number,prompt,dataset,labels):

    pos_res, neg_res = [], []

    # 用于匹配的特征：
    features_match = None
    # 文本
    if prompt[1]:
      # str -> torch.Tensor [1,11] / [CLS] + len(prompt) + [SEP]
      text = text_tokenizer(f"{prompt}", return_tensors='pt', padding=True)['input_ids'].to("cuda")
      with torch.no_grad():
        # torch.Tensor [1,11] -> torch.Tensor [1,768] / [Batch_size, features_length]
        text_features = text_encoder(text).logits
        text_features = text_features / torch.linalg.norm(text_features, dim=1)
        features_match = text_features if features_match is None else features_match + text_features

    # 图像
    image_data = data["images"]
    ref_image_range = range(data["info"][positive_cate][0],data["info"][positive_cate][0]+image_number)
    for i in ref_image_range:
      # print(image_data[i]["file_name"])   # JPEG 常规集图片
      image = processor(images=Image.open(image_data[i]["file_path"]), return_tensors="pt").to("cuda")
      with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=image["pixel_values"])
        image_features = image_features / torch.linalg.norm(image_features, dim=1)
        features_match = image_features if features_match is None else features_match + image_features

    # [1,768]
    features_match = (features_match / torch.linalg.norm(features_match, dim=1)).to("cuda")

    pos_list, neg_list = [], []
    for cate_name in data["info"]:
        if cate_name == positive_cate:
            index = data["info"][cate_name]
            pos_list.append(range(index[0],index[dataset]))
        else:
            index = data["info"][cate_name]
            neg_list.append(range(index[0],index[dataset]))
    pos_list = [
    x for r in pos_list for x in r
    if x not in ref_image_range
    ]
    neg_list = [x for r in neg_list for x in r]

    if not set(pos_list) & set(neg_list):
        print(f"pos_list和neg_list互斥，分别包含的图片数为{len(pos_list)},{len(neg_list)}")
    else:
        print("pos_list和neg_list有交集")
        print(set(pos_list) & set(neg_list))

    for m in range(len(data["images"])):
        image_features = torch.tensor(data["images"][m][labels]).to("cuda")
        if m in pos_list:
            with torch.no_grad():
                # clip里计算logits是文本和图像余弦相似度乘以exp(logit_scale)
                logits_per_image = (clip_model.logit_scale.exp()) * image_features @ features_match.t()
                # print(logits_per_image)
                pos_res.append(torch.max(logits_per_image).item())
                # print(pos_res)
        if m in neg_list:
            with torch.no_grad():
                logits_per_image = (clip_model.logit_scale.exp()) * image_features @ features_match.t()
                neg_res.append(torch.max(logits_per_image).item())
    print(len(pos_res))
    print(len(neg_res))
    return pos_res, neg_res

def eval(pos_res, neg_res, threshold):

    p, n = np.array(pos_res), np.array(neg_res)
    tp = sum(p >= threshold)
    fp = sum(n >= threshold)
    fn = sum(p < threshold)
    # tn = sum(res_neg < threshold)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score, precision, recall

def main(postive_cate,image_number,prompt,dataset,labels):
    # 获取所有样本的clip相似度分数
    pos_res,neg_res = clip_by_threshold(postive_cate,image_number,prompt,dataset,labels)
    # print(len(pos_res),len(neg_res))         # 20/40,980/1960

    cate_name = postive_cate

    # Top-K precision / K=1,5,10,20
    K=[1,5,10,20]
    labels = [1] * len(pos_res) + [0] * len(neg_res)
    scores = list(pos_res) + list(neg_res)
    all_items = list(zip(scores, labels))
    sorted_items = sorted(all_items, key=lambda x: x[0], reverse=True)
    k_precision = []
    for k in K:
        top_k = sorted_items[:k]
        num_pos_in_top_k = sum([label for score, label in top_k])
        k_precision.append(num_pos_in_top_k / k)
    print(f"{cate_name} Top-{K} precision:", k_precision)

    # 阈值范围：[全局最小logit，全局最大logit]
    thresholds = np.linspace(min(min(pos_res), min(neg_res)), max(max(pos_res), max(neg_res)), 200)

    best_threshold, best_f1_score, best_precision, best_recall  = 0., 0., 0., 0.
    f1_scores = []

    for threshold in thresholds:
        f1_score, precision, recall = eval(pos_res, neg_res, threshold)
        f1_scores.append(f1_score)

        # 判断最佳f1
        if f1_score > best_f1_score:
            # 更新指标
            best_threshold, best_f1_score, best_precision, best_recall = threshold, f1_score, precision, recall

    print(f"{cate_name}_best_threshold", best_threshold)
    print(f"{cate_name}_best_precision", best_precision)
    print(f"{cate_name}_best_recall", best_recall)
    print(f"{cate_name}_best_f1_score", best_f1_score)

    return k_precision, [best_threshold, best_precision, best_recall, best_f1_score]

# 五类正样本
pos_table = ["chair", 'cup', 'bicycle', 'skateboard', 'oven']
pos_table_cn = ["椅子", '杯子', '自行车', '滑板', '烤箱']

f1 = [[0] * 4 for _ in range(5)]
k_pre = [[0] * 4 for _ in range(5)]

for i in range(0, 5):
  print(pos_table[i])
  prompt1 = pos_table_cn[i]
  prompt2 = "这是一张"+pos_table_cn[i]+"的照片"
  k_pre[i], f1[i] = main(postive_cate=pos_table[i],image_number=image_number,prompt=[prompt2,with_prompt], dataset=dataset, labels=labels)
labels_k_pre = ['K=1','K=5','K=10','K=20']
k_pre.append(np.mean(k_pre, axis=0))
k_pre.insert(0, labels_k_pre)
labels_f1 = ["Best Threshold","Precision","Recall","F1-Score"]
f1.append(np.mean(f1, axis=0))
f1.insert(0, labels_f1)

print(f'Qwen/{labels}/{dataset_type[dataset - 1]}/是否微调:{if_lora}/是否加入文本:{with_prompt}/参考图n={image_number}')

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

# path1 = './Qwen_Labels.json'
# path2 = './Qwenlora_Labels.json'
# with open(path1, "r", encoding="utf-8") as f:
#   labels1 = json.load(f)
# with open(path2, "r", encoding="utf-8") as f:
#   labels2 = json.load(f)
