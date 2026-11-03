"""Build an offline HTML replay from recorded outputs; never changes ranking."""
import argparse
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="docs/demo/results.json")
    parser.add_argument("--output", default="docs/demo/index.html")
    args = parser.parse_args()
    report = json.loads(Path(args.results).read_text(encoding="utf-8"))
    sources = json.loads((ROOT / "examples/showcase/sources.json").read_text(encoding="utf-8"))
    template = (Path(__file__).parent / "page.html").read_text(encoding="utf-8")
    data = json.dumps({"report": report, "sources": sources}, ensure_ascii=False).replace("<", "\\u003c")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root = Path(os.path.relpath(ROOT / "examples/showcase", output.parent)).as_posix() + "/"
    guide = Path(os.path.relpath(ROOT / "docs/DEMO.md", output.parent)).as_posix()
    page = template.replace("/*__RECORDED_DATA__*/", data)
    page = page.replace("/*__IMAGE_ROOT__*/", json.dumps(image_root))
    page = page.replace("__GUIDE_PATH__", html.escape(guide, quote=True))
    output.write_text(page, encoding="utf-8")
    raw_target = output.parent / "results.json"
    if raw_target.resolve() != Path(args.results).resolve():
        raw_target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
