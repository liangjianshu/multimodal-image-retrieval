"""Resolve search intent without replacing explicit user constraints with a summary."""

PRESERVE_INTENT_PROMPT = '''你是图片检索请求解析器。提取目标和条件，不要把完整需求缩成物体类别。
规则：
1. 必须保留用户指定的颜色、材质、形状、场景、位置、关系、数量、否定条件和修改要求。
   description 描述完整目标；target_category 只填主要类别，不能用类别替代完整目标。
2. 用户文字优先于参考图。参考图用于识别主体或补充未指定的外观；不能用参考图里的背景、颜色覆盖用户要修改的条件。
3. constraints 逐项摘录用户的限制条件，否定词必须一起保留；没有明确条件时为 []。不得猜测用户未提出的限制。
4. 只有参考图片而没有文字时，按相似图片检索处理，description 写可见主体与主要外观，constraints 为 []。
5. 只有用户明确要求返回几张图片时，top_k 才填正整数；物体数量不是返回数量，没有要求时填 null。
   问候、知识问答、询问图片内容、只要求描述图片且不找其他图片，都不是图片检索。
   “杯子”“白色自行车”这样的目标短语在本搜索工具中视为检索；不要因为句子短就拒绝。
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

CATEGORY_ROUTING_PROMPT = '''
补充任务：根据 available_categories（图库模型已识别的类别），为检索选择候选类别。
在上面的 JSON 中额外返回 category_filter 和 candidate_categories：
- category_filter="selected"：candidate_categories 是从 available_categories 原样选取的类别列表。
  允许选择多个同义、上下位或功能相近的类别，以免名称粒度差异造成漏检；尊重用户明确的类别排除要求。
  类别初筛只判断主体大类，颜色、材质、背景等细节交给后续原图排序，不要因此删除整个大类。
  若目标明确但图库没有相关类别，返回 selected 和空列表，不要把无关类别硬凑进去。
- category_filter="all"：目标类别不明确，或用户明确跨品类找风格时不限制类别，candidate_categories=[]。
  all 不是普通检索的默认值。已识别出明确主体且 available_categories 有相应类别时必须用 selected。
  例如输入类别中已有“动物”，用户要求“白色动物”，颜色不确定也必须选“动物”，不能因此选 all。
  使用 all 时 target_category 留空，表示没有单一明确的目标类别。
- needs_retrieval=false 时，category_filter="none"、candidate_categories=[]。
不能创造 available_categories 以外的新名称。只有图片而没有文字时也要调用此规则，按相似图片检索处理。
不要输出图片编号、排名或相似度；这里只判断意图与类别范围。
'''

CATEGORY_GROUP_PROMPT = '''你负责统一图片检索图库的主体类别名称。
输入是模型已识别的类别列表。返回一个 JSON 对象，key 是每个原类别，value 是其代表类别。
规则：
1. 每个输入类别必须恰好出现一次，value 必须从输入类别中选择；代表类别映射到自己。
2. 把同义、上下位和用途相近且适合相似物品检索的类别合并，例如“鞋/运动鞋”、“盘子/碟子”。
3. 只合并同一具体物品家族，不因为同色、同材质或同属“物品/家具/器具”等宽泛概念就合并。
4. 有明确较宽类别时用较宽类别作代表；不同名称都可作相似用途候选时，选择一个代表统一分组。
5. 不确定是否同一物品家族时保持原类别。不能删除任何类别，不能输出解释或图片编号。
类别列表：
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
