# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
from qwen_vl_utils import process_vision_info
import torch, json, re, random, os
from peft import PeftModel
import numpy as np
from PIL import Image
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import CLIPProcessor, CLIPModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from modelscope import AutoModelForCausalLM, AutoTokenizer

# ===== 配置 =====
category_feature_key = "category_feature"  # 类别特征（第一阶段：筛选）
desc_feature_key = "Qwen_features2"        # 描述特征（第二阶段：精排）
dataset = 2  # 常规=1, 常规+细粒度=2

# 类别筛选阈值：取类别最高分的比例，筛出同类图片
CATEGORY_FILTER_RATIO = 0.85
# 描述精排阈值：在筛选结果中，取描述最高分的比例
DESC_RANK_RATIO = 0.85

# ===== 加载图像特征数据 =====
features_path = './Qwenlora_Labels_features.json'
with open(features_path, "r", encoding="utf-8") as f:
    image_db = json.load(f)

# ===== 加载模型 =====
print("加载 Taiyi-CLIP text encoder...")
text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
text_encoder = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval().to("cuda")

print("加载 CLIP image encoder...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

print("加载 Qwen2.5-VL 意图识别模型（图搜图）...")
qwen_model_path = "/path/to/workspace/multimodel_project/model-Qwen2.5-VL-7B-Instruct"
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_model_path, torch_dtype=torch.bfloat16, device_map="cuda:1", trust_remote_code=True
)
# TODO: 加载 LoRA adapter，将下面路径替换为你的 adapter 路径
lora_path = "/path/to/workspace/multimodel_project/project_three/output_lora_llm"
qwen_model = PeftModel.from_pretrained(qwen_model, lora_path)
qwen_processor = AutoProcessor.from_pretrained(qwen_model_path, trust_remote_code=True)

print("加载 Qwen2.5 纯文本意图识别模型（文搜图/不需要检索）...")
qwen_text_model_path = "/path/to/workspace/multimodel_project/model-Qwen2.5-7B-Instruct"
qwen_text_model = AutoModelForCausalLM.from_pretrained(
    qwen_text_model_path, device_map="cuda:0", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
)
qwen_text_tokenizer = AutoTokenizer.from_pretrained(qwen_text_model_path)

print("所有模型加载完成\n")

# ===== 意图识别 Prompt =====
INTENT_PROMPT = '''你的任务是为用户需求分类；

分类依据包含四点：第一，是否需要图像检索，只能从下面提供的【是否图像检索】中选择；
第二、如果需要图像检索，只能从下面提供的【问题类型】中选择；
第三、如果需要图像检索，用简短的语句描述用户想要检索的目标物体；
第四、如果需要图像检索，则包括检索描述以及检索方式是什么，检索方式从下面提供的【检索方式】中选择；

只能从以下提供的列表中选择每一项的结果：
【问题】： 和用户提出的需求一致，根据需求给出下面四个选项的答案
【是否需要图像检索】 ： 1或者0，1和0分别代表需要图像检索和不需要图像检索
【问题类型】：["通过文字提取需求检索图片","通过图片提取需求检索图片",""]
【检索描述】：用简短的语句描述用户想检索的目标物体，例如"红色的椅子"、"公路自行车"、"咖啡杯"等。没有明确检索目标则为[""]
【检索方式】：["卡阈值", ["TopK",K], ""]

一些"通过文字提取需求检索图片"的例子：
{"问题": ["请展示三张椅子的图片。"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索描述": ["椅子"], "检索方式": ["TopK",3]}
{"问题": ["帮我找四张杯子的照片"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索描述": ["杯子"], "检索方式": ["TopK",4]}
{"问题": ["能不能发几张滑板的图片？"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索描述": ["滑板"], "检索方式": ["卡阈值"]}
{"问题": ["帮我找几张红色的自行车图片"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索描述": ["红色的自行车"], "检索方式": ["卡阈值"]}

一些"通过图片提取需求检索图片"的例子：
注意：当用户上传了图片时，即使文字中没有明确提及类别，你也必须根据图片内容识别出物体并用简短语句描述。只有当图片中没有明确物体或者没上传图片时，【检索描述】才为[""]。
{"问题": ["帮我找出与上传图片最相似的一张图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索描述": ["根据图片内容描述目标物体"], "检索方式": ["TopK",1]}
{"问题": ["请帮我找几张和这张图片风格相似的椅子图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索描述": ["椅子"], "检索方式": ["卡阈值"]}

一些"不需要图像检索的例子"：
对于不需要图像检索的类型，【问题类型】【检索描述】【检索方式】都是[""]，也即{"是否需要图像检索": 0, "问题类型": [""], "检索描述": [""], "检索方式": [""]}
{"问题": ["请问中国的首都是哪里？"], "是否需要图像检索": 0, "问题类型": [""], "检索描述": [""], "检索方式": [""]}

"TopK"主要针对问题中出现明确数量要求的情况，比如["{n}张"]，{n}是任意数字（包括中文数字和阿拉伯数字），K需要根据问题来确定具体数字，且输出为["TopK",K]。中文数字需要转换为阿拉伯数字，例如：一=1,二=2,三=3,四=4,五=5,六=6,七=7,八=8,九=9,十=10。比如以下例子：
{"问题": ["我需要三张烤箱的照片"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索描述": ["烤箱"], "检索方式": ["TopK",3]}

"卡阈值"主要针对问题中不明确提及数量的情况，比如["几张","一些","所有"],检索方式为["卡阈值"]，比如以下例子：
{"问题": ["能不能发几张滑板的图片？"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索描述": ["滑板"], "检索方式": ["卡阈值"]}

按照以下格式输出，保证输出的字符可以直接转换为json：
{"问题": [""], "是否需要图像检索": [""], "问题类型": [""], "检索描述": [""], "检索方式": [""]}

严格按照以上的要求和格式对需求分类，不要生成多余内容：
'''


# ===== 意图识别函数 =====
def qwen_vl_inference(user_query, image_path=None):
    """对单条用户指令做意图识别，返回结构化字典。有图片用VL模型，无图片用纯文本模型。"""
    if image_path:
        # 有图片：用 VL 模型
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": INTENT_PROMPT + "问题：" + user_query},
        ]}]

        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda:1")

        generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print("[使用 Qwen2.5-VL 模型（图搜图）]")
    else:
        # 无图片：用纯文本 Qwen 模型
        question = INTENT_PROMPT + "问题：" + user_query
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        text = qwen_text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = qwen_text_tokenizer([text], return_tensors="pt").to(qwen_text_model.device)
        generated_ids = qwen_text_model.generate(**model_inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = qwen_text_tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        print("[使用 Qwen2.5 纯文本模型（文搜图/不需要检索）]")

    # 解析 JSON
    match = re.search(r'\{.*\}', response, re.S)
    if not match:
        print(f"[意图识别失败] 无法匹配JSON: {response[:100]}")
        return None

    raw = match.group()
    raw = re.sub(r'(?<=,)\s*([A-Za-z_]\w*)\s*=\s*([^,\]\}]+)', r' "\1=\2"', raw)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[意图识别失败] JSON解析失败: {raw[:120]}")
        return None


# ===== 构建查询特征 =====
def build_query_features(query_text, ref_image_path=None):
    """将查询文本和/或参考图片编码为特征向量"""
    features_match = None

    # 文本特征
    if query_text:
        text_ids = text_tokenizer(query_text, return_tensors='pt', padding=True)['input_ids'].to("cuda")
        with torch.no_grad():
            text_features = text_encoder(text_ids).logits
            text_features = text_features / torch.linalg.norm(text_features, dim=1, keepdim=True)
            features_match = text_features

    # 参考图像特征（图搜图时）
    if ref_image_path:
        image_input = clip_processor(images=Image.open(ref_image_path), return_tensors="pt").to("cuda")
        with torch.no_grad():
            img_feat = clip_model.get_image_features(pixel_values=image_input["pixel_values"])
            img_feat = img_feat / torch.linalg.norm(img_feat, dim=1, keepdim=True)
            features_match = img_feat if features_match is None else features_match + img_feat

    if features_match is None:
        return None

    return (features_match / torch.linalg.norm(features_match, dim=1)).to("cuda")


# ===== 两阶段检索函数 =====
def two_stage_search(query_text, ref_image_path=None):
    """
    第一阶段：用类别特征筛选同类图片
    第二阶段：用描述特征在同类中精排
    返回: [(image_index, score, file_path), ...] 按描述分数降序排列
    """
    query_feat = build_query_features(query_text, ref_image_path)
    if query_feat is None:
        print("错误：无法构建查询特征")
        return []

    # ===== 第一阶段：类别筛选 =====
    category_scores = []
    for idx in range(len(image_db["images"])):
        cat_feat = torch.tensor(image_db["images"][idx][category_feature_key]).to("cuda")
        with torch.no_grad():
            score = (clip_model.logit_scale.exp() * cat_feat @ query_feat.t()).max().item()
        category_scores.append((idx, score))

    # 按类别分数筛选
    max_cat_score = max(s for _, s in category_scores)
    cat_threshold = max_cat_score * CATEGORY_FILTER_RATIO
    filtered_indices = [idx for idx, score in category_scores if score >= cat_threshold]

    print(f"  [阶段1] 类别筛选: 最高分={max_cat_score:.2f}, 阈值={cat_threshold:.2f}, 筛出{len(filtered_indices)}张")

    # ===== 第二阶段：描述精排 =====
    results = []
    for idx in filtered_indices:
        desc_feat = torch.tensor(image_db["images"][idx][desc_feature_key]).to("cuda")
        with torch.no_grad():
            score = (clip_model.logit_scale.exp() * desc_feat @ query_feat.t()).max().item()
        results.append((idx, score, image_db["images"][idx]["file_path"]))

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"  [阶段2] 描述精排: 最高分={results[0][1]:.2f}" if results else "  [阶段2] 无结果")
    return results


# ===== 结果展示 =====
def display_topk(results, k):
    """展示 Top-K 结果"""
    print(f"\n{'='*50}")
    print(f"Top-{k} 检索结果")
    print(f"{'='*50}")
    for i, (idx, score, path) in enumerate(results[:k]):
        print(f"  [{i+1}] score={score:.2f}  {path}")


def display_threshold(results, threshold, max_display=5):
    """卡阈值展示：从超过阈值的图片中随机选几张"""
    filtered = [(idx, score, path) for idx, score, path in results if score >= threshold]
    if not filtered:
        print(f"\n没有图片的相似度超过阈值 {threshold:.1f}")
        return
    selected = random.sample(filtered, min(max_display, len(filtered)))
    print(f"\n{'='*50}")
    print(f"卡阈值检索结果 (阈值={threshold:.1f}, 共{len(filtered)}张超过阈值, 随机展示{len(selected)}张)")
    print(f"{'='*50}")
    for i, (idx, score, path) in enumerate(selected):
        print(f"  [{i+1}] score={score:.2f}  {path}")


# ===== 解析意图字段的辅助函数 =====
def parse_list_field(field):
    """从意图结果中提取列表字段的第一个元素"""
    if isinstance(field, list):
        return field[0] if field else ""
    return field if field else ""


# ===== 交互式主循环 =====
def interactive():
    print("\n" + "=" * 50)
    print("  多模态图像检索系统")
    print("  输入检索指令，输入 quit 退出")
    print("=" * 50)

    while True:
        user_input = input("\n请输入检索指令（可附带图片路径）: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("退出系统")
            break
        if not user_input:
            continue

        # 从输入中正则提取图片路径
        ref_image_path = None
        path_match = re.search(r'(/\S+\.(?:jpg|jpeg|png|bmp|webp))', user_input, re.IGNORECASE)
        if path_match:
            ref_image_path = path_match.group(1)
            if not os.path.exists(ref_image_path):
                print(f"图片路径不存在: {ref_image_path}")
                continue
            print(f"检测到参考图片: {ref_image_path}")

        # ===== Step 1: 意图识别（文字+图片一次性传入） =====
        print("\n[Step 1] 正在识别意图...")
        intent = qwen_vl_inference(user_input, image_path=ref_image_path)
        if intent is None:
            print("意图识别失败，请重新输入")
            continue

        print(f"意图识别结果: {json.dumps(intent, ensure_ascii=False)}")

        # 判断是否需要图像检索
        if intent.get("是否需要图像检索") == 0:
            print("该问题不需要图像检索")
            continue

        # 解析检索描述
        search_desc = parse_list_field(intent.get("检索描述", ""))

        # 解析检索方式
        search_method = intent.get("检索方式", [""])

        # ===== Step 2: 两阶段检索 =====
        # 构建查询文本：有描述用描述，否则用原始输入
        query_text = search_desc if search_desc else user_input

        print(f"\n[Step 2] 正在检索... 描述: {search_desc or '无'}, 查询: {query_text}")
        results = two_stage_search(query_text, ref_image_path)

        if not results:
            print("未找到任何结果")
            continue

        # ===== Step 3: 按检索方式输出结果 =====
        print(f"\n[Step 3] 输出检索结果...")

        if isinstance(search_method, list) and len(search_method) >= 2 and search_method[0] == "TopK":
            # TopK 模式
            k = int(search_method[1])
            display_topk(results, k)
        else:
            # 卡阈值模式（在精排结果中取相对阈值）
            max_score = results[0][1] if results else 0
            threshold = max_score * DESC_RANK_RATIO
            display_threshold(results, threshold)


if __name__ == "__main__":
    try:
        interactive()
    except KeyboardInterrupt:
        print("\n用户中断，退出系统")
