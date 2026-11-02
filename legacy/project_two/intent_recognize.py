# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
import json, os, re, sys
import argparse

# ======================== 阶段1: GPT生成意图识别数据集 ========================
def stage1_gpt_generate(output_path="./gpt_intent.json", num_rounds=200):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")

    prompt = '''
    问题模拟用户提问；

    答案内容包含四点：第一，是否需要图像检索，从【是否图像检索】中选择；
    第二、如果需要图像检索，判断只根据问题文本内容做检索还是需要结合图像来检索，从【问题类型】选择；
    第三、如果需要图像检索，判断需要检索的类别是什么，从【类别】中选择；
    第四、如果需要图像检索，则包括检索的类别以及检索方式是什么，检索方式从【检索方式】中选择。
    最终的输出是 json 格式，具体为参考为：
    {"问题": [""], "是否需要图像检索": "";"问题类型": [""]; "检索类别": [""]; "检索方式": [""]}

    【类别】：["椅子","杯子","自行车","滑板","烤箱",""]
    【是否图像检索】 ： [1, 0]，1和0分别代表需要图像检索和不需要图像检索
    【问题类型】：["文搜图","图搜图",""]
    【检索方式】：["卡阈值", "TopK",""]

    如果是TopK，则K需要根据问题来确定具体数字，且输出为["TopK",K]；

    当问题出现没有明确指定数量的情况，则检索方式为"卡阈值"；

    不同问题类型例子：
    {"问题": ["给我两张自行车的图片？"], "是否需要图像检索": 1, "问题类型": ["文搜图"], "检索类别": ["自行车"], "检索方式": ["TopK",2]}
    {"问题": ["有没有和这张图片中款式相同的烤箱？"], "是否需要图像检索": 1, "问题类型": ["图搜图"], "检索类别": ["烤箱"], "检索方式": ["卡阈值"]}
    {"问题": ["给我一张类似的图片。"], "是否需要图像检索": 1, "问题类型": ["图搜图"], "检索类别": [""], "检索方式": ["TopK",1]}
    {"问题": ["今天天气怎么样？"], "是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}

    提问方式尽可能多样，句式可以进行变换；

    生5个不需要图像检索的句子随机问题句子, 放在同一个列表中[{},{},...]
    '''

    def get_gpt_intent(prompt):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        full_prompt = prompt
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "你是一个图像检索问答对的设计者，严格按照json文件格式输出"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=32768,
            temperature=0.85
        )

        return response.choices[0].message.content

    intent_data = []
    seen_questions = set()
    for i in range(num_rounds):
        print(f"GPT生成第 {i+1}/{num_rounds} 轮...")
        result = get_gpt_intent(prompt)
        intent_data += json.loads(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(intent_data, f, ensure_ascii=False, indent=2)
    print(f"GPT数据已保存到 {output_path}，共 {len(intent_data)} 条")


# ======================== 阶段2: Qwen2.5 文本意图识别 ========================
def stage2_qwen_text_inference(
    gpt_intent_path="./gpt_intent.json",
    model_path="/path/to/workspace/multimodel_project/model-Qwen2.5-7B-Instruct",
    output_path="./Qwen_7B_intent.json",
    label_index=0
):
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    import torch, time

    print(f"加载Qwen2.5模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda:1", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(gpt_intent_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 兼容两种格式：列表(stage1生成) 或 字典(已分组)
    if isinstance(raw_data, list):
        data = {"文搜图": [], "图搜图": [], "不需要图像检索": []}
        for item in raw_data:
            qtype = item.get("问题类型", [""])[0] if isinstance(item.get("问题类型"), list) else item.get("问题类型", "")
            if qtype == "文搜图":
                data["文搜图"].append(item)
            elif qtype == "图搜图":
                data["图搜图"].append(item)
            else:
                data["不需要图像检索"].append(item)
    else:
        data = raw_data

    def get_response(question):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    prompt = '''
        你的任务是为用户需求分类；

        分类依据包含四点：第一，是否需要图像检索，只能从下面提供的【是否图像检索】中选择；
        第二、如果需要图像检索，只能从下面提供的【问题类型】中选择；
        第三、如果需要图像检索，判断需要检索的类别是什么，只能从下面提供的【检索类别】中选择；
        第四、如果需要图像检索，则包括检索的类别以及检索方式是什么，检索方式从下面提供的【检索方式】中选择；

        只能从以下提供的列表中选择每一项的结果：
        【问题】： 和用户提出的需求一致，根据需求给出下面四个选项的答案
        【是否需要图像检索】 ： 1或者0，1和0分别代表需要图像检索和不需要图像检索
        【问题类型】：["通过文字提取需求检索图片","通过图片提取需求检索图片",""]
        【检索类别】：["椅子","杯子","自行车","滑板","烤箱","其他类别"]
        【检索方式】：["卡阈值", ["TopK",K], ""]

        对于【检索方式】，"TopK"表示按相似度排序给出最高的K张图片；"卡阈值"表示按照要求设定某个相似度阈值，相似度超过阈值的图片则认为满足条件。

        当问题中出现"一些"，"几张"和"所有"等数量不明确的词，检索方式是"卡阈值"；

        如果是TopK，则K需要根据问题来确定具体数字，且输出为["TopK",K]；比如"给我一张自行车的图片"，则{"检索方式":["TopK",K=1]}；

        一些"通过文字提取需求检索图片"的例子：
        {"问题": ["请展示三张椅子的图片。"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["椅子"], "检索方式": ["TopK",3]}
        {"问题": ["能不能发几张滑板的图片？"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["滑板"], "检索方式": ["卡阈值"]}
        {"问题": ["请给我所有烤箱的照片。"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["烤箱"], "检索方式": ["卡阈值"]}
        {"问题": ["找一些滑板相关的图片。"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["滑板"], "检索方式": ["卡阈值"]}

        一些"通过图片提取需求检索图片"的例子：
        {"问题": ["帮我找出与上传图片最相似的一张图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索类别": [""], "检索方式": ["TopK",1]}
        {"问题": ["请帮我找几张和这张图片风格相似的椅子图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索类别": [""], "检索方式": ["卡阈值"]}
        {"问题": ["找出和这张烤箱图片最接近的一些图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索类别": ["烤箱"], "检索方式": ["卡阈值"]}
        {"问题": ["请用这张图片为参考，找出所有类似的自行车图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索类别": ["自行车"], "检索方式": ["卡阈值"]}
        {"问题": ["有两张不同型号的烤箱图片吗？"],"是否需要图像检索": 1,"问题类型": ["文搜图"],"检索类别": ["烤箱"],"检索方式": ["TopK",2]},

        一些"不需要图像检索的例子"：
        对于不需要图像检索的类型，【问题类型】【检索类别】【检索方式】都是[""]，也即{"是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}
        输出格式为：
        {"问题": ["请问中国的首都是哪里？"], "是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}
        {"问题": ["你能告诉我今天星期几吗？"], "是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}
        {"问题": ["北京的天气预报是什么？"], "是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}
        {"问题": ["你知道世界上最高的山峰叫什么吗？"], "是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}

        按照以下格式输出：
        {"问题": "", "是否需要图像检索": "", "问题类型": "", "检索类别": "", "检索方式": ""}
        特别注意！！非常严格的标准，对于检索方式的区别，当明确指定所需要的数量的时候，如一张，两张等带有明确数字的我们采用topk检索，其他一律采用卡阈值，想有几张这种都属于卡阈值

        严格按照以上的要求和格式对需求分类，不要生成多余内容：
        '''

    data_intent = {"文搜图": [], "图搜图": [], "不需要图像检索": []}
    label_list = ["文搜图", "图搜图", "不需要图像检索"]
    label_list_en =["text", "image", "other"]
    label = label_list[label_index]

    # 检查是否已有该类别的推理结果
    output_path = f"./Qwen_7B_intent_{label_list_en[label_index]}.json"
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict) and len(existing.get(label, [])) > 0:
            print(f"[跳过] {output_path} 中已有 {label} 的推理结果 ({len(existing[label])}条)")
            return
        data_intent = existing  # 保留其他类别已有的结果

    start = time.time()
    for i in range(len(data[label])):
        question = prompt + "问题：" + data[label][i]["问题"][0]
        resp = get_response(question)
        match = re.search(r'\{[^}]*\}', resp, re.S)
        if match:
            data_intent[label].append(json.loads(match.group()))
        else:
            print(f"[WARN] 第{i}条无法解析: {resp[:100]}")
        if i % 100 == 0:
            print(f"{i}/{len(data[label])}, 耗时: {time.time()-start:.1f}s")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_intent, f, ensure_ascii=False, indent=2)
    print(f"Qwen推理结果已保存到 {output_path}")


# ======================== 阶段3: Qwen2.5-VL 多模态意图识别 ========================
def stage3_qwen_vl_inference(
    gpt_intent_path="./gpt_intent.json",
    image_json_path="./imagenet_json.json",
    model_path="/path/to/workspace/multimodel_project/model-Qwen2.5-VL-7B-Instruct",
    label_index=0
):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch, random, time

    # 中英文类别映射
    CN_TO_EN = {"椅子": "chair", "杯子": "cup", "自行车": "bicycle", "滑板": "skateboard", "烤箱": "oven"}

    print(f"加载Qwen2.5-VL模型: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:1", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    with open(image_json_path, "r", encoding="utf-8") as f:
        image_data = json.load(f)
    with open(gpt_intent_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 兼容两种格式：列表(stage1生成) 或 字典(已分组)
    if isinstance(raw_data, list):
        data = {"文搜图": [], "图搜图": [], "不需要图像检索": []}
        for item in raw_data:
            qtype = item.get("问题类型", [""])[0] if isinstance(item.get("问题类型"), list) else item.get("问题类型", "")
            if qtype == "文搜图":
                data["文搜图"].append(item)
            elif qtype == "图搜图":
                data["图搜图"].append(item)
            else:
                data["不需要图像检索"].append(item)
    else:
        data = raw_data

    # 构建图片路径索引（用英文key访问）
    cate_name_list = ["椅子", "杯子", "自行车", "滑板", "烤箱"]
    image_list = {}
    for cn_name in cate_name_list:
        en_name = CN_TO_EN[cn_name]
        image_list[cn_name] = []
        info = image_data["info"][en_name]
        for i in range(info[0], info[1]):
            image_list[cn_name].append(image_data["images"][i]["file_path"])

    image_path_total = [img["file_path"] for img in image_data["images"]]

    def get_image_path(cate_name):
        if cate_name == [] or cate_name == "":
            return random.choice(image_path_total)
        if cate_name in cate_name_list:
            return random.choice(image_list[cate_name])
        return random.choice(image_path_total)

    # Qwen-VL
    prompt = '''
    你的任务是为用户需求分类；

    分类依据包含四点：第一，是否需要图像检索，只能从下面提供的【是否图像检索】中选择；
    第二、如果需要图像检索，只能从下面提供的【问题类型】中选择；
    第三、如果需要图像检索，判断需要检索的类别是什么，只能从下面提供的【检索类别】中选择；
    第四、如果需要图像检索，则包括检索的类别以及检索方式是什么，检索方式从下面提供的【检索方式】中选择；

    只能从以下提供的列表中选择每一项的结果：
    【问题】： 和用户提出的需求一致，根据需求给出下面四个选项的答案
    【是否需要图像检索】 ： 1或者0，1和0分别代表需要图像检索和不需要图像检索
    【问题类型】：["通过文字提取需求检索图片","通过图片提取需求检索图片",""]
    【检索类别】：["椅子","杯子","自行车","滑板","烤箱",""]，没有明确提及的类别则为[""]
    【检索方式】：["卡阈值", ["TopK",K], ""]

    一些"通过文字提取需求检索图片"的例子：
    {"问题": ["请展示三张椅子的图片。"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["椅子"], "检索方式": ["TopK",3]}
    {"问题": ["能不能发几张滑板的图片？"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["滑板"], "检索方式": ["卡阈值"]}
    {"问题": ["请给我所有烤箱的照片。"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["烤箱"], "检索方式": ["卡阈值"]}

    一些"通过图片提取需求检索图片"的例子：
    {"问题": ["帮我找出与上传图片最相似的一张图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索类别": [""], "检索方式": ["TopK",1]}
    {"问题": ["请帮我找几张和这张图片风格相似的椅子图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索类别": ["椅子"], "检索方式": ["卡阈值"]}
    {"问题": ["找出和这张烤箱图片最接近的一些图片。"], "是否需要图像检索": 1, "问题类型": ["通过图片提取需求检索图片"], "检索类别": ["烤箱"], "检索方式": ["卡阈值"]}

    一些"不需要图像检索的例子"：
    对于不需要图像检索的类型，【问题类型】【检索类别】【检索方式】都是[""]，也即{"是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}
    {"问题": ["请问中国的首都是哪里？"], "是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}
    {"问题": ["你能告诉我今天星期几吗？"], "是否需要图像检索": 0, "问题类型": [""], "检索类别": [""], "检索方式": [""]}

    "TopK"主要针对问题中出现明确数量要求的情况，比如["{n}张"]，{n}是任意[数字]，K需要根据问题来确定具体数字，且输出为["TopK",K]；比如以下例子：
    {'问题': ['帮我找出和这张图片风格最接近的一张图片。'], '是否需要图像检索': 1, '问题类型': ['图搜图'], '检索类别': [], '检索方式': ['TopK', 1]}
    {"问题": ["我需要三张烤箱的照片"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["烤箱"], "检索方式": ["TopK",3]}

    "卡阈值"主要针对问题中不明确提及数量的情况，比如["几张","一些","所有"],检索方式为["卡阈值"]，比如以下例子：
    {"问题": ["能不能发几张滑板的图片？"], "是否需要图像检索": 1, "问题类型": ["通过文字提取需求检索图片"], "检索类别": ["滑板"], "检索方式": ["卡阈值"]}
    {'问题': ['请检索与上传图片风格相同的滑板。'], '是否需要图像检索': 1, '问题类型': ['图搜图'], '检索类别': ['滑板'], '检索方式': ['卡阈值']}
    {'问题': ['有没有和这张照片里的杯子一样的款式？'], '是否需要图像检索': 1, '问题类型': ['图搜图'], '检索类别': ['杯子'], '检索方式': ['卡阈值']}
    {'问题': ['请检索与这张滑板图片外观相近的图片。'], '是否需要图像检索': 1, '问题类型': ['图搜图'], '检索类别': ['滑板'], '检索方式': ['卡阈值']}

    按照以下格式输出，保证输出的字符可以直接转换为json：
    {"问题": [""], "是否需要图像检索": [""], "问题类型": [""], "检索类别": [""], "检索方式": [""]}

    严格按照以上的要求和格式对需求分类，不要生成多余内容：
    '''

    def get_messages(question, qcate, scate):
        if qcate in [["文搜图"], ["图搜图"]]:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": get_image_path(scate[0] if scate != [] else scate)},
                    {"type": "text", "text": prompt + "问题：" + question},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt + "问题：" + question}],
            }]
        return messages

    def get_vl_response(messages):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda:1")
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    data_intent = {"文搜图": [], "图搜图": [], "不需要图像检索": []}
    label_list = ["文搜图", "图搜图", "不需要图像检索"]
    label_list_en =["text", "image", "other"]
    label = label_list[label_index]

    # 检查是否已有该类别的推理结果
    output_path = f"./Qwen_VL_7B_intent_{label_list_en[label_index]}.json"
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict) and len(existing.get(label, [])) > 0:
            print(f"[跳过] {output_path} 中已有 {label} 的推理结果 ({len(existing[label])}条)")
            return
        data_intent = existing

    start = time.time()
    for i in range(len(data[label])):
        q = data[label][i]
        response = get_vl_response(get_messages(q["问题"][0], q["问题类型"], q["检索类别"]))
        # 用贪婪匹配，从第一个 { 到最后一个 }
        match = re.search(r'\{.*\}', response[0], re.S)
        if match:
            raw = match.group()
            # 修复模型常见的非法 JSON：K=3 -> "K=3", 去掉尾部多余逗号
            raw = re.sub(r'(?<=,)\s*([A-Za-z_]\w*)\s*=\s*([^,\]\}]+)', r' "\1=\2"', raw)
            raw = re.sub(r',\s*([}\]])', r'\1', raw)
            try:
                data_intent[label].append(json.loads(raw))
            except json.JSONDecodeError:
                print(f"[WARN] 第{i}条JSON解析失败: {raw[:120]}")
        else:
            print(f"[WARN] 第{i}条无法匹配JSON: {response[0][:100]}")
        if i % 100 == 0:
            print(f"{i}/{len(data[label])}, 耗时: {time.time()-start:.1f}s")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_intent, f, ensure_ascii=False, indent=2)
    print(f"Qwen-VL推理结果已保存到 {output_path}")


# ======================== 阶段4: 评估 ========================
def stage4_evaluate(
    gpt_intent_path="./gpt_intent.json",
    pred_path=None,
    model_name="Qwen"
):
    """对比模型推理结果和GPT ground truth，计算各维度Accuracy"""

    # 加载GT
    with open(gpt_intent_path, "r", encoding="utf-8") as f:
        raw_gt = json.load(f)

    # GT分组
    if isinstance(raw_gt, list):
        gt = {"文搜图": [], "图搜图": [], "不需要图像检索": []}
        for item in raw_gt:
            qtype = item.get("问题类型", [""])[0] if isinstance(item.get("问题类型"), list) else item.get("问题类型", "")
            if qtype == "文搜图":
                gt["文搜图"].append(item)
            elif qtype == "图搜图":
                gt["图搜图"].append(item)
            else:
                gt["不需要图像检索"].append(item)
    else:
        gt = raw_gt

    # 加载预测结果
    with open(pred_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    # GT中"是否需要图像检索"的映射：文搜图/图搜图→1，不需要→0
    GT_NEED_SEARCH = {"文搜图": 1, "图搜图": 1, "不需要图像检索": 0}

    label_list = ["文搜图", "图搜图", "不需要图像检索"]
    all_results = {}

    print(f"\n{'='*60}")
    print(f"  评估模型: {model_name} ({pred_path})")
    print(f"{'='*60}")

    total_correct = {"是否检索": 0, "检索类别": 0, "检索方式": 0}
    total_count = 0

    for label in label_list:
        gt_items = gt.get(label, [])
        pred_items = pred.get(label, [])

        if not pred_items:
            print(f"\n[{label}] 无推理结果，跳过")
            continue

        n = min(len(gt_items), len(pred_items))
        correct = {"是否检索": 0, "检索类别": 0, "检索方式": 0}

        for i in range(n):
            g, p = gt_items[i], pred_items[i]

            # 1. 是否需要图像检索
            gt_need = GT_NEED_SEARCH[label]
            pred_need = p.get("是否需要图像检索", "")
            # 兼容字符串和数字
            if isinstance(pred_need, list):
                pred_need = pred_need[0] if pred_need else ""
            try:
                pred_need = int(pred_need)
            except (ValueError, TypeError):
                pred_need = -1
            if pred_need == gt_need:
                correct["是否检索"] += 1
            else:
                question = g.get("问题", [""])[0] if isinstance(g.get("问题"), list) else g.get("问题", "")
                print(f"  ✗ [{label}] 第{i}条 是否检索 错误 | 问题: {question}")
                print(f"    正确: {gt_need}  预测: {pred_need}")

            # 2. 检索类别
            gt_cate = g.get("检索类别", [""])
            pred_cate = p.get("检索类别", [""])
            if not isinstance(gt_cate, list):
                gt_cate = [gt_cate]
            if not isinstance(pred_cate, list):
                pred_cate = [pred_cate]
            # 归一化：[] 和 [''] 视为等价
            if gt_cate == [""]:
                gt_cate = []
            if pred_cate == [""]:
                pred_cate = []
            if set(gt_cate) == set(pred_cate):
                correct["检索类别"] += 1
            else:
                question = g.get("问题", [""])[0] if isinstance(g.get("问题"), list) else g.get("问题", "")
                print(f"  ✗ [{label}] 第{i}条 检索类别 错误 | 问题: {question}")
                print(f"    正确: {gt_cate}  预测: {pred_cate}")

            # 3. 检索方式
            gt_method = g.get("检索方式", [""])
            pred_method = p.get("检索方式", [""])
            if not isinstance(gt_method, list):
                gt_method = [gt_method]
            if not isinstance(pred_method, list):
                pred_method = [pred_method]
            # 归一化：[] 和 [''] 视为等价
            if gt_method == [""]:
                gt_method = []
            if pred_method == [""]:
                pred_method = []
            # 统一转字符串比较（处理 ["TopK", 3] vs ["TopK", "3"]）
            if [str(x) for x in gt_method] == [str(x) for x in pred_method]:
                correct["检索方式"] += 1
            else:
                question = g.get("问题", [""])[0] if isinstance(g.get("问题"), list) else g.get("问题", "")
                print(f"  ✗ [{label}] 第{i}条 检索方式 错误 | 问题: {question}")
                print(f"    正确: {gt_method}  预测: {pred_method}")

        # 打印该类别结果
        print(f"\n[{label}] (共{n}条)")
        for key in correct:
            acc = correct[key] / n * 100 if n > 0 else 0
            print(f"  {key}: {correct[key]}/{n} = {acc:.1f}%")

        all_results[label] = {k: f"{v}/{n} ({v/n*100:.1f}%)" for k, v in correct.items()}

        for k in total_correct:
            total_correct[k] += correct[k]
        total_count += n

    # 总体结果
    if total_count > 0:
        print(f"\n[总体] (共{total_count}条)")
        for key in total_correct:
            acc = total_correct[key] / total_count * 100
            print(f"  {key}: {total_correct[key]}/{total_count} = {acc:.1f}%")
        all_results["总体"] = {k: f"{v}/{total_count} ({v/total_count*100:.1f}%)" for k, v in total_correct.items()}

    # 保存评估结果
    eval_output = pred_path.replace(".json", "_eval.json")
    with open(eval_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到 {eval_output}")


# ======================== 主入口 ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="意图识别Pipeline")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4],
                        help="1=GPT生成数据, 2=Qwen文本推理, 3=Qwen-VL多模态推理, 4=评估")
    parser.add_argument("--label_index", type=int, default=0,
                        help="类别索引: 0=文搜图, 1=图搜图, 2=不需要图像检索")
    parser.add_argument("--pred_path", type=str, default=None,
                        help="评估用: 推理结果文件路径")
    parser.add_argument("--model_name", type=str, default="Qwen",
                        help="评估用: 模型名称")
    args = parser.parse_args()

    if args.stage == 1:
        stage1_gpt_generate()
    elif args.stage == 2:
        stage2_qwen_text_inference(label_index=args.label_index)
    elif args.stage == 3:
        stage3_qwen_vl_inference(label_index=args.label_index)
    elif args.stage == 4:
        if not args.pred_path:
            print("请指定 --pred_path 推理结果文件")
            sys.exit(1)
        stage4_evaluate(pred_path=args.pred_path, model_name=args.model_name)

# python intent_recognize.py --stage 4 --model_name Qwen --pred_path /path/to/workspace/multimodel_project/project_two/Qwen_7B_intent_text.json
# python intent_recognize.py --stage 4 --model_name Qwen --pred_path /path/to/workspace/multimodel_project/project_two/Qwen_7B_intent_other.json
# python intent_recognize.py --stage 4 --model_name Qwen-VL --pred_path /path/to/workspace/multimodel_project/project_two/Qwen_VL_7B_intent_image.json
# python intent_recognize.py --stage 4 --model_name Qwen-VL --pred_path /path/to/workspace/multimodel_project/project_two/Qwen_VL_7B_intent_other.json
# python intent_recognize.py --stage 4 --model_name Qwen-VL --pred_path /path/to/workspace/multimodel_project/project_two/Qwen_VL_7B_intent_text.json
