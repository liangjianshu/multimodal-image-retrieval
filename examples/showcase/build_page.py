"""Build an offline HTML replay from recorded outputs; never changes ranking."""
import argparse
import base64
import html
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
GUIDE_URL = "https://github.com/liangjianshu/multimodal-image-retrieval/blob/main/docs/DEMO.md"


def initial_view(report, sources, images):
    """Render the first result even in previews that do not execute JavaScript."""
    escape = html.escape
    query = report["queries"][0]
    by_id = {source["id"]: source for source in sources}
    inputs = ""
    if query.get("image"):
        inputs += f'<img class="query-photo" src="{images[query["image"]]}" alt="查询参考图片">'
    if query.get("text"):
        inputs += f'<div class="prompt">{escape(query["text"])}</div>'
    cards = []
    for number, result in enumerate(query["results"], 1):
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
    if len(query["results"]) < 3:
        cards.append(f'<div class="empty">类别粗筛只保留了 {query["candidate_count"]} 张候选图片。</div>')
    return {
        "__INITIAL_LABEL__": {"text": "TEXT → IMAGE", "image": "IMAGE → IMAGE", "fusion": "IMAGE + TEXT → IMAGE"}[query["kind"]],
        "__INITIAL_TITLE__": escape(query["title"]),
        "__INITIAL_INPUT__": inputs,
        "__INITIAL_NOTE__": {"text": "查询文本直接编码，不调用意图解析。", "image": "仅使用参考图向量；已排除图库中的同一张图片。", "fusion": "图片与文本向量各占 50%；已排除参考图自身。"}[query["kind"]],
        "__INITIAL_PIPELINE__": "CLIP 图像编码 → 图像相似度排序" if query["mode"] == "clip" else "查询编码 → 类别粗筛 → 描述排序",
        "__INITIAL_COUNT__": f' / {len(query["results"])} 张',
        "__INITIAL_CANDIDATES__": f'图库 {report["gallery_size"]} 张 · {"图像候选" if query["mode"] == "clip" else "粗筛候选"} {query["candidate_count"]} 张 · Top-3',
        "__INITIAL_CARDS__": "".join(cards),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="docs/demo/results.json")
    parser.add_argument("--output", default="docs/demo/index.html")
    parser.add_argument("--image-mode", choices=("embedded", "external"), default="embedded",
                        help="Embed images for a portable file, or copy assets for fast web loading")
    args = parser.parse_args()
    report = json.loads(Path(args.results).read_text(encoding="utf-8"))
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
    data = json.dumps({"report": report, "sources": sources, "images": images}, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("/*__RECORDED_DATA__*/", data)
    page = page.replace("__GUIDE_PATH__", GUIDE_URL)
    raw = base64.b64encode(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")).decode("ascii")
    page = page.replace("__RAW_RESULTS_URL__", "data:application/json;base64," + raw)
    for marker, value in initial_view(report, sources, images).items():
        page = page.replace(marker, value)
    output.write_text(page, encoding="utf-8")
    raw_target = output.parent / "results.json"
    if raw_target.resolve() != Path(args.results).resolve():
        raw_target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
