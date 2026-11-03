"""Lazy model loading: data tools and the smoke demo never download weights."""
import json

import numpy as np

DEFAULTS = {
    "backend": "qwen-semantic",
    "embedding_model": "google/embeddinggemma-2",
    "embedding_revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
    "embedding_weights_sha256": None, "embedding_model_path": None, "embedding_dimension": 768,
    "vision_tokens": 280, "embedding_batch_size": 4,
    "text_encoder": "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese",
    "image_encoder": "openai/clip-vit-large-patch14",
    "vl_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "text_model": "Qwen/Qwen2.5-3B-Instruct",
    "adapter": None, "device": "auto", "max_pixels": 262144,
    "max_new_tokens": 384, "category_ratio": 0.85, "image_weight": 0.5,
    "prompt_profile": "historical",
    "intent_profile": "preserve",
}


def resolve_device(device):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "npu" or device.startswith("npu:"):
        # Optional Ascend backend; install the vendor-matched torch/torch_npu pair.
        import torch_npu  # noqa: F401
    return device


def extract_json(text):
    """Accept fenced JSON without executing model-generated text."""
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character in "{[":
            try:
                value, _ = decoder.raw_decode(text[position:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Model did not return valid JSON: {text[:120]!r}")


class Encoders:
    """Taiyi text + its documented CLIP ViT-L/14 image space (768 dimensions)."""
    def __init__(self, config):
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer, CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = resolve_device(config["device"])
        self.text_tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
        self.text_model = BertForSequenceClassification.from_pretrained(config["text_encoder"]).eval().to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained(config["image_encoder"])
        self.image_model = CLIPModel.from_pretrained(config["image_encoder"]).eval().to(self.device)

    def text(self, text):
        inputs = self.text_tokenizer(text, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[-1] > 512:
            raise ValueError("Text exceeds the encoder's 512-token limit; shorten it explicitly instead of silently dropping constraints")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            result = self.text_model(**inputs).logits[0]
        return result.float().cpu().numpy()

    def image(self, path):
        from PIL import Image
        with Image.open(path) as image:
            inputs = self.image_processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.image_model.get_image_features(**inputs)
            # Transformers 4 returns a tensor; 5 returns projected pooler_output.
            result = (output.pooler_output if hasattr(output, "pooler_output") else output)[0]
        return result.float().cpu().numpy()


class Qwen:
    def __init__(self, config, vision=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        self.torch, self.vision, self.config = torch, vision, config
        self.device = resolve_device(config["device"])
        use_cuda = self.device.startswith("cuda")
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_cuda else torch.float32
        if self.device.startswith("npu"):
            dtype = torch.bfloat16
        model_id = config["vl_model"] if vision else config["text_model"]
        cls = Qwen2_5_VLForConditionalGeneration if vision else AutoModelForCausalLM
        self.model = cls.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa").to(self.device)
        if vision:
            self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=config["max_pixels"])
            if config.get("adapter"):
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, config["adapter"])
        else:
            self.processor = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()

    def generate(self, prompt, image=None):
        if self.vision:
            from qwen_vl_utils import process_vision_info
            content = [{"type": "text", "text": prompt}]
            if image is not None:
                content.insert(0, {"type": "image", "image": str(image)})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        else:
            messages = [{"role": "user", "content": prompt}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False)
        return self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def describe(self, path):
        if self.config.get("prompt_profile") == "historical":
            from .historical_prompts import CATEGORY_PROMPT, DESCRIPTION_PROMPT
            result = {"categories": extract_json(self.generate(CATEGORY_PROMPT, path)),
                      "description": self.generate(DESCRIPTION_PROMPT, path).strip()}
        else:
            result = extract_json(self.generate(
                '请描述图片。只返回 JSON 对象，格式为 {"categories":["主要物体类别"],'
                '"description":"中文描述，包括可见的颜色、形状、材质和场景"}。不要猜测不可见属性。', path))
        if not isinstance(result, dict) or not isinstance(result.get("categories"), list) or not result["categories"]:
            raise ValueError("Description response requires nonempty categories")
        if any(not isinstance(c, str) or not c.strip() for c in result["categories"]):
            raise ValueError("Invalid generated category")
        if not isinstance(result.get("description"), str) or not result["description"].strip():
            raise ValueError("Description response requires description text")
        return result

    def category_groups(self, categories):
        from .intent import CATEGORY_GROUP_PROMPT
        mapping = extract_json(self.generate(CATEGORY_GROUP_PROMPT + json.dumps(categories, ensure_ascii=False)))
        if not isinstance(mapping, dict) or set(mapping) != set(categories):
            raise ValueError("Category normalization must cover every gallery label")
        if any(not isinstance(v, str) or v not in categories for v in mapping.values()):
            raise ValueError("Category normalization invented a label")
        if any(mapping[v] != v for v in mapping.values()):
            raise ValueError("Category representatives must map to themselves")
        return mapping

    def intent(self, text, image=None, *, available_categories=None, category_aliases=None, parse_image_intent=False):
        if self.config.get("intent_profile", "preserve") == "preserve":
            if not parse_image_intent and available_categories is None and image is not None and not text.strip():
                # An image-only request in this search tool already specifies the operation.
                visual = self.describe(image)
                return {"needs_retrieval": True, "target_category": "、".join(visual["categories"]),
                        "description": visual["description"], "constraints": [], "top_k": None,
                        "source": "image-description"}
            from .intent import PRESERVE_INTENT_PROMPT, CATEGORY_ROUTING_PROMPT, ROUTE_INTENT_PROMPT
            payload = {"text": text, "has_reference_image": image is not None}
            prompt = ROUTE_INTENT_PROMPT if parse_image_intent else PRESERVE_INTENT_PROMPT
            if available_categories is not None:
                payload["available_categories"] = available_categories
                prompt += CATEGORY_ROUTING_PROMPT + "\n用户输入（JSON 数据）：\n"
            result = extract_json(self.generate(prompt + json.dumps(payload, ensure_ascii=False), image))
            if not isinstance(result, dict) or type(result.get("needs_retrieval")) is not bool:
                raise ValueError("Intent response requires boolean needs_retrieval")
            if not result["needs_retrieval"]:
                # A negative routing decision is complete. Retrieval-only metadata
                # may be omitted/null and must never block or trigger a search.
                stopped = {"needs_retrieval": False, "target_category": "", "description": "",
                           "constraints": [], "top_k": None}
                if available_categories is not None:
                    stopped.update(category_filter="none", candidate_categories=[], raw_response=result)
                if parse_image_intent:
                    stopped["raw_response"] = result
                return stopped
            if any(not isinstance(result.get(key), str) for key in ("description", "target_category")):
                raise ValueError("Intent response requires text description and target_category")
            constraints = result.get("constraints")
            if not isinstance(constraints, list) or any(not isinstance(c, str) or not c.strip() for c in constraints):
                raise ValueError("Intent constraints must be a list of nonempty strings")
            k = result.get("top_k")
            if k is not None and (type(k) is not int or not 1 <= k <= 1000):
                raise ValueError("Intent top_k must be null or an integer from 1 to 1000")
            if available_categories is not None:
                # Preserve the exact model fields before schema defaults/operation rules.
                raw_response = dict(result)
                aliases = category_aliases or {}
                recognized = aliases.get(result["target_category"], result["target_category"])
                if isinstance(result.get("candidate_categories"), list):
                    result["candidate_categories"] = [aliases.get(c, c) if isinstance(c, str) else c
                                                      for c in result["candidate_categories"]]
                if recognized in available_categories and (
                    result.get("category_filter") in (None, "all") or "candidate_categories" not in result
                ):
                    # A known, model-recognized target supplies an unambiguous filter
                    # if the model omits the additional routing schema. No label is invented.
                    result.update(category_filter="selected", candidate_categories=[recognized],
                                  category_resolution="exact-recognized-category")
                if image is not None and not text.strip():
                    # Visible attributes are not explicit user constraints.
                    result["constraints"] = []
                mode, selected = result.get("category_filter"), result.get("candidate_categories")
                if mode not in ("selected", "all", "none") or not isinstance(selected, list):
                    raise ValueError("Intent requires category_filter and candidate_categories")
                if any(not isinstance(c, str) or c not in available_categories for c in selected):
                    raise ValueError("Intent selected an unknown gallery category")
                if mode == "none" or (mode == "all" and selected):
                    raise ValueError("Invalid category filter for retrieval")
                result["candidate_categories"] = list(dict.fromkeys(selected))
                result["raw_response"] = raw_response
            if parse_image_intent and available_categories is None:
                result["raw_response"] = dict(result)
            return result
        if self.config.get("prompt_profile") == "historical":
            from .historical_prompts import INTENT_PROMPT
            raw = extract_json(self.generate(INTENT_PROMPT + "问题：" + text, image))
            if not isinstance(raw, dict):
                raise ValueError("Intent response must be an object")
            def scalar(value):
                return value[0] if isinstance(value, list) and value else value
            needed = scalar(raw.get("是否需要图像检索"))
            description = scalar(raw.get("检索描述", ""))
            method = raw.get("检索方式", [])
            if needed not in (0, 1, "0", "1") or not isinstance(description, str):
                raise ValueError("Invalid historical intent response")
            k = method[1] if isinstance(method, list) and len(method) > 1 and method[0] == "TopK" else None
            if k is not None and (type(k) is not int or not 1 <= k <= 1000):
                raise ValueError("Invalid historical intent top_k")
            return {"needs_retrieval": needed in (1, "1"), "description": description,
                    "top_k": k, "raw_response": raw}
        result = extract_json(self.generate(
            '将用户需求解析为图像检索请求，只返回 JSON：'
            '{"needs_retrieval":true,"description":"中文检索目标与属性","top_k":5}。'
            '非图像检索请求将 needs_retrieval 设为 false；没有明确数量时 top_k 为 null。'
            '如提供参考图片，根据可见内容提取检索目标。\n用户需求：' + text, image))
        if not isinstance(result, dict) or not isinstance(result.get("needs_retrieval"), bool):
            raise ValueError("Intent response requires boolean needs_retrieval")
        if not isinstance(result.get("description", ""), str):
            raise ValueError("Intent description must be text")
        k = result.get("top_k")
        if k is not None and (type(k) is not int or not 1 <= k <= 1000):
            raise ValueError("Intent top_k must be null or an integer from 1 to 1000")
        return result
