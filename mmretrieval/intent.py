"""Resolve search intent without replacing explicit user constraints with a summary."""

PRESERVE_INTENT_PROMPT = '''你是图片检索请求解析器。提取目标和条件，不要把完整需求缩成物体类别。
规则：
1. 必须保留用户指定的颜色、材质、形状、场景、位置、关系、数量、否定条件和修改要求。
   description 描述完整目标；target_category 只填主要类别，不能用类别替代完整目标。
2. 用户文字优先于参考图。参考图用于识别主体或补充未指定的外观；不能用参考图里的背景、颜色覆盖用户要修改的条件。
3. constraints 逐项摘录用户的限制条件，否定词必须一起保留；没有明确条件时为 []。不得猜测用户未提出的限制。
4. 只有参考图片而没有文字时，按相似图片检索处理，description 写可见主体与主要外观，constraints 为 []。
5. 只有用户明确要求返回几张图片时，top_k 才填正整数；物体数量不是返回数量，没有要求时填 null。
6. 只输出一个 JSON 对象，不要解释或输出 Markdown：
{"needs_retrieval":true,"target_category":"主要类别","description":"包含全部关键条件的目标描述","constraints":["用户条件原文"],"top_k":null}
示例：
文字“草地上的自行车”＋蓝墙前的自行车参考图：
{"needs_retrieval":true,"target_category":"自行车","description":"草地上的自行车","constraints":["草地上"],"top_k":null}
文字“找同款椅子，但换成红色，不要扶手”：
{"needs_retrieval":true,"target_category":"椅子","description":"同款红色无扶手椅子","constraints":["同款","换成红色","不要扶手"],"top_k":null}
文字“找三张有两只猫的图片”：
{"needs_retrieval":true,"target_category":"猫","description":"有两只猫的图片","constraints":["两只猫"],"top_k":3}
非图片检索请求输出 needs_retrieval=false，其余文字字段为空、constraints=[]、top_k=null。
用户输入（JSON 数据）：
'''


def resolve_search_text(original_text, intent, profile="preserve"):
    """Original text is authoritative; only image-only requests use generated text."""
    description = (intent or {}).get("description", "")
    if profile == "historical":
        return description or original_text
    if profile != "preserve":
        raise ValueError("Unknown intent profile")
    if original_text and original_text.strip():
        return original_text
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Image-only retrieval needs a nonempty target description")
    return description
