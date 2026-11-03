"""Build an offline HTML replay from recorded outputs; never changes ranking."""
import argparse
import base64
import html
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
GUIDE_URL = "https://github.com/liangjianshu/multimodal-image-retrieval/blob/main/docs/DEMO.md"


def initial_view(report, sources, images, min_score=None):
    """Render the first result even in previews that do not execute JavaScript."""
    escape = html.escape
    query = report["queries"][0]
    visible = [r for r in query["results"] if min_score is None or r["score"] >= min_score]
    by_id = {source["id"]: source for source in sources}
    inputs = ""
    if query.get("image"):
        inputs += f'<img class="query-photo" src="{images[query["image"]]}" alt="查询参考图片">'
    if query.get("text"):
        inputs += f'<div class="prompt">{escape(query["text"])}</div>'
    cards = []
    for number, result in enumerate(visible, 1):
        categories = escape(" · ".join(result["retrieval_categories"]))
        cards.append(
            '<article class="card"><div class="photo-wrap"><button class="photo-button" '
            f'aria-label="查看 {escape(result["id"])} 的图片描述">'
            f'<img src="{images[result["file_path"]]}" alt="{categories}"></button>'
            f'<span class="rank">{number:02d}</span></div><div class="card-body">'
            f'<div class="card-title">{categories}</div><div class="score">'
            f'<span>余弦相似度</span><b>{result["score"]:.3f}</b></div>'
            f'<p class="description">{escape(result["retrieval_description"])}</p>'
            f'<a class="source" href="{escape(by_id[result["id"]]["source"], quote=True)}">'
            f'{escape(result["id"])} · Wikimedia / CC0 ↗</a></div></article>'
        )
    if query.get("status") == "not_retrieval":
        cards.append('<div class="empty">Qwen 判定这条请求不需要图片检索，已跳过向量编码和搜索。</div>')
    elif query.get("status") == "no_category_match":
        cards.append('<div class="empty">图库没有匹配的目标类别，未返回无关图片。</div>')
    elif not visible:
        cards.append('<div class="empty">当前请求没有可返回的图片。</div>' if min_score is None else
                     '<div class="empty">暂无达到当前相似度要求的结果。试试降低阈值。</div>')
    elif len(visible) < len(query["results"]):
        cards.append(f'<div class="empty">已过滤 {len(query["results"]) - len(visible)} 张低于阈值 {min_score:.2f} 的结果。</div>')
    elif len(visible) < min(query.get("top_k") or 3, 3):
        cards.append(f'<div class="empty">当前可用候选为 {query["candidate_count"]} 张图片。</div>')
    note = {"text": "查询文本直接编码，不调用意图解析。", "image": "仅使用参考图向量；已排除图库中的同一张图片。", "fusion": "图片与文本向量各占 50%；已排除参考图自身。", "intent": "先判断是否需要图片检索。"}[query["kind"]]
    pipeline = "CLIP 图像编码 → 图像相似度排序" if query["mode"] == "clip" else "查询编码 → 类别粗筛 → 描述排序"
    if query.get("intent"):
        note = ('实际编码文字（保留原文）：' if query.get('text') else '模型提取的图片目标：') + (query.get('search_text') or query['intent'].get('description', '')) + '。文字与参考图向量各占 50%；已排除参考图自身。'
        pipeline = "需求解析（保留用户原文）→ 图文融合 → 类别粗筛 → 描述排序"
    semantic = report.get("backend") == "qwen-semantic"
    routed = report.get("backend") in ("qwen-semantic", "qwen-embedding")
    hybrid = report.get("backend") == "qwen-gemma2"
    direct = report.get("backend") in ("embeddinggemma2", "qwen-gemma2", "qwen-embedding")
    if routed:
        intent = query["intent"]
        note = "Qwen 已确认需要检索。" + (f"用户要求返回 {intent['top_k']} 张。" if intent.get('top_k') else "") if intent['needs_retrieval'] else "Qwen 判定：无需图片检索。已跳过检索。"
        pipeline = ("Qwen 意图识别 → 类别粗筛 → 描述向量排序" if semantic else "Qwen 意图识别 → 原图向量直接排序") if intent['needs_retrieval'] else "Qwen 意图识别 → 不触发检索"
    elif hybrid:
        intent = query["intent"]
        scope = "、".join(intent["candidate_categories"]) if intent["category_filter"] == "selected" else "全图库"
        note = f'Qwen 识别目标：{intent["target_category"]}；候选类别：{scope}。原始文字与参考图直接编码。' if intent["needs_retrieval"] else "Qwen 判定：无需图片检索。已跳过向量编码和搜索。"
        pipeline = "Qwen 意图识别 → 类别粗筛 → EmbeddingGemma 2 原图排序" if intent["needs_retrieval"] else "Qwen 意图识别 → 不触发检索"
    elif direct:
        note = {"text": "完整文字直接编码，与图库原图向量比较。", "image": "参考图像素直接编码；已排除参考图自身。", "fusion": "完整文字与参考图联合编码；已排除参考图自身。"}[query["kind"]]
        pipeline = "原图 / 完整文字 → EmbeddingGemma 2 → 图库图像向量排序"
    return {
        "__INITIAL_LABEL__": {"text": "TEXT → IMAGE", "image": "IMAGE → IMAGE", "fusion": "IMAGE + TEXT → IMAGE", "intent": "QWEN → ROUTING"}[query["kind"]],
        "__INITIAL_TITLE__": escape(query["title"]),
        "__INITIAL_INPUT__": inputs,
        "__INITIAL_NOTE__": escape(note),
        "__INITIAL_PIPELINE__": pipeline,
        "__INITIAL_COUNT__": f' / {len(visible)} 张',
        "__INITIAL_CANDIDATES__": "未执行检索" if query.get("status") == "not_retrieval" else f'图库 {report["gallery_size"]} 张 · {"粗筛候选" if hybrid else "图像候选" if direct or query["mode"] == "clip" else "粗筛候选"} {query["candidate_count"]} 张 · 最多 {min(query.get("top_k") or 3, 3)} 张',
        "__INITIAL_CARDS__": "".join(cards),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="docs/demo/results.json")
    parser.add_argument("--output", default="docs/demo/index.html")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Optional replay threshold; disabled by default so category-filtered Top-K is shown")
    parser.add_argument("--image-mode", choices=("embedded", "external"), default="embedded",
                        help="Embed images for a portable file, or copy assets for fast web loading")
    args = parser.parse_args()
    report = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if args.min_score is not None and (not math.isfinite(args.min_score) or not 0 <= args.min_score <= 1):
        raise ValueError("Showcase min-score must be between 0 and 1")
    min_score = args.min_score if report.get("backend") in ("embeddinggemma2", "qwen-gemma2", "qwen-embedding") else None
    sources = json.loads((ROOT / "examples/showcase/sources.json").read_text(encoding="utf-8"))
    template = (Path(__file__).parent / "page.html").read_text(encoding="utf-8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root = ROOT / "examples/showcase"
    images = {}
    for source in sources:
        image_path = (image_root / source["file_path"]).resolve()
        if not image_path.is_relative_to(image_root.resolve()):
            raise ValueError("Image path must stay inside the showcase directory")
        if args.image_mode == "embedded":
            images[source["file_path"]] = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
        else:
            target = output.parent / "images" / image_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != image_path:
                shutil.copyfile(image_path, target)
            images[source["file_path"]] = "images/" + image_path.name
    data = json.dumps({"report": report, "sources": sources, "images": images,
                       "display": {"min_score": min_score}}, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("/*__RECORDED_DATA__*/", data)
    adapted = bool(report.get("adapter_verification"))
    semantic = report.get("backend") == "qwen-semantic"
    routed = report.get("backend") in ("qwen-semantic", "qwen-embedding")
    hybrid = report.get("backend") == "qwen-gemma2"
    direct = report.get("backend") in ("embeddinggemma2", "qwen-gemma2", "qwen-embedding")
    if routed:
        badge = "<b>Qwen2.5-VL-7B</b> · 意图识别 / 原 LoRA"
        method = "路线 A · 类别 + 描述检索" if semantic else "路线 B · 原图向量检索"
        intro = "先由 Qwen 判断检索意图，再识别类别、筛选候选，按图片描述的向量排序。" if semantic else "先由 Qwen 判断检索意图，再直接编码完整文字与原图，不使用类别粗筛。"
        model_note = "路线 A：Qwen 原 LoRA 生成类别与描述，Taiyi 编码文字；图文查询结合 CLIP 参考图向量。" if semantic else ("路线 B：EmbeddingGemma 2 已完成类别对比学习试训；这组演示不证明细粒度属性提升。" if report.get('training_state') == 'finetuned' else "路线 B 对照：当前展示未微调的公开权重基线。")
        if routed and not semantic:
            model_note += ' <a href="https://liangjianshu.github.io/multimodal-image-retrieval/route-b-baseline/">微调前对照</a> · <a href="https://github.com/liangjianshu/multimodal-image-retrieval/blob/main/docs/EMBEDDING_EXPERIMENT.md">试训报告</a>'
    elif hybrid:
        model_name = report["config"]["vl_model"].split("/")[-1].replace("-Instruct", "")
        badge = f'<b>{html.escape(model_name)}</b> · ' + ("作者原 LoRA" if adapted else "公开基座")
        method = "EmbeddingGemma 2 · 原图排序"
        intro = "先判断是否需要检索，再确定候选类别。结合完整文字和参考图，在自己的图库中寻找目标。"
        model_note = ("Qwen 原 LoRA" if adapted else "Qwen 公开基座") + "用于类别生成、意图识别与候选类别选择；EmbeddingGemma 2 负责原图排序。12 张图的定性演示不代表规模化准确率。"
    elif direct:
        badge = "<b>EmbeddingGemma 2</b> · 原图编码"
        method = "文字 / 图片 / 图文联合编码"
        intro = "一句中文、一张参考图，或两者结合。用同一个多模态编码器，直接检索自己的图片库。"
        model_note = "使用公开编码器；图片说明沿用原微调 Qwen 的描述，仅供阅读，不参与排序。12 张图的定性演示不代表规模化准确率。"
    else:
        model_name = report["config"]["vl_model"].split("/")[-1].replace("-Instruct", "")
        badge = f'<b>{html.escape(model_name)}</b> ' + ("作者原 LoRA" if adapted else "公开基座")
        method = "目标解析 + 两阶段语义检索" if adapted else "语义检索 + CLIP 图像检索"
        intro = "一句中文、一张参考图，或两者结合。用 Qwen 理解图片，以 Taiyi-CLIP 检索自己的图库。"
        model_note = "已加载作者原 LoRA；小图库定性演示，不代表规模化准确率或历史 F1 收益。" if adapted else "公开基座的小图库定性演示；未加载 LoRA，不代表规模化准确率或历史 F1 收益。"
    page = page.replace("__MODEL_BADGE__", badge).replace("__METHOD_BADGE__", method)
    page = page.replace("__INTRO__", intro)
    route_nav = ''
    if routed:
        base = 'https://liangjianshu.github.io/multimodal-image-retrieval/'
        route_nav = ('<nav class="route-nav" aria-label="检索路线">'
            f'<a href="{base}" aria-current="{str(semantic).lower()}">路线 A · Qwen 类别与描述</a>'
            f'<a href="{base}route-b/" aria-current="{str(not semantic).lower()}">路线 B · 原图 Embedding</a></nav>')
    page = page.replace('__ROUTE_NAV__', route_nav)

    page = page.replace("__INTENT_TAB__", '<button role="tab" data-kind="intent">非检索请求</button>' if hybrid or routed else '')
    nonretrieval_examples = (
        '<section class="example-group" aria-labelledby="nonretrieval-examples-title">'
        '<h2 id="nonretrieval-examples-title">非检索示例</h2>'
        '<p>识别问答、图片描述等请求，不启动图片搜索。</p>'
        '<div class="presets" id="nonretrieval-presets"></div></section>'
    ) if any(q.get("status") == "not_retrieval" for q in report["queries"]) else ''
    page = page.replace("__NONRETRIEVAL_EXAMPLES__", nonretrieval_examples)
    controls = '' if min_score is None else (
        '<div class="filter-controls" aria-label="结果相似度过滤">'
        '<label for="min-score">最低相似度</label>'
        f'<input id="min-score" type="range" min="0" max="1" step="0.01" value="{min_score:.2f}" '
        f'aria-describedby="filter-hint"><output id="min-score-value" for="min-score">{min_score:.2f}</output>'
        '<button id="clear-filter" type="button">关闭分数阈值</button>'
        '<span id="filter-hint">阈值越高，返回越少；最多展示三张</span></div>'
    )
    page = page.replace("__FILTER_CONTROLS__", controls)
    page = page.replace("__DESCRIPTION_NOTE__", "Qwen 类别用于粗筛，图片描述参与向量排序。" if semantic else "图片说明仅供阅读；类别不参与筛选。" if routed else "Qwen 类别参与粗筛，详细说明仅供阅读。" if hybrid else "图片说明仅供阅读，不参与排序。" if direct else "点击图片可查看完整描述。")
    page = page.replace("__MODEL_NOTE__", model_note)
    page = page.replace("__GUIDE_PATH__", GUIDE_URL)
    raw = base64.b64encode(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")).decode("ascii")
    page = page.replace("__RAW_RESULTS_URL__", "data:application/json;base64," + raw)
    for marker, value in initial_view(report, sources, images, min_score).items():
        page = page.replace(marker, value)
    output.write_text(page, encoding="utf-8")
    raw_target = output.parent / "results.json"
    if raw_target.resolve() != Path(args.results).resolve():
        raw_target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
