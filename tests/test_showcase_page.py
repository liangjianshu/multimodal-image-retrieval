"""Regression: the demo must show result images without adjacent files or JS."""
import base64
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class StaticImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img" and attrs.get("src"):
            self.sources.append(attrs["src"])


def test_standalone_page_contains_real_initial_images(tmp_path):
    output = tmp_path / "standalone.html"
    results_path = ROOT / "docs/demo/results.json"
    subprocess.run([
        sys.executable, str(ROOT / "examples/showcase/build_page.py"),
        "--results", str(results_path), "--output", str(output),
    ], check=True, capture_output=True)
    page = output.read_text(encoding="utf-8")
    report = json.loads(results_path.read_text(encoding="utf-8"))
    first_query = report["queries"][0]
    expected_results = [r for r in first_query["results"]
                        if report.get("backend") not in ("embeddinggemma2", "qwen-gemma2") or r["score"] >= 0.66]
    parser = StaticImages()
    parser.feed(page)
    assert len(parser.sources) == len(expected_results)
    for src, result in zip(parser.sources, expected_results):
        assert src.startswith("data:image/jpeg;base64,")
        expected = (ROOT / "examples/showcase" / result["file_path"]).read_bytes()
        assert base64.b64decode(src.split(",", 1)[1]) == expected
    assert first_query["title"] in page.split("<script>", 1)[0]
