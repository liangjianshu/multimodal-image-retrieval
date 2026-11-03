"""Fetch the small CC0 gallery and retain source URLs and checksums.

Images are already bundled. Run this only when intentionally refreshing the gallery.
"""
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
TITLES = [
    "Red chair.jpg", "Chair.JPG", "Office Chair.jpg", "Chair from wood veneer.jpg",
    "Coffeecup.jpg", "Coffee cup seen from above.jpg", "Coffee cup kaffe - 3.JPG",
    "Coffee cup from Turkey, Ottoman period, late 19th-early 20th century, silver, porcelain.JPG",
    "White bicycle.jpg", "A shared bicycle in the grass.jpg",
    "Rustic Bicycle Against Blue Door.jpg", "Bicycle 02.jpg",
]
HEADERS = {"User-Agent": "MultimodalRetrievalDemo/0.1 (https://github.com/liangjianshu/multimodal-image-retrieval)"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return response.read()


def main():
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    sources, catalog = [], []
    for number, title in enumerate(TITLES, 1):
        params = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "imageinfo",
            "iiprop": "url|extmetadata", "iiurlwidth": 720, "titles": "File:" + title})
        data = json.loads(get("https://commons.wikimedia.org/w/api.php?" + params))
        info = next(iter(data["query"]["pages"].values()))["imageinfo"][0]
        metadata = info["extmetadata"]
        license_name = metadata["LicenseShortName"]["value"]
        if license_name != "CC0":
            raise ValueError(f"Unexpected license for {title}: {license_name}")
        image_id = f"image-{number:02d}"
        relative = f"images/{image_id}.jpg"
        url = info.get("thumburl", info["url"])
        payload = get(url)
        (ROOT / relative).write_bytes(payload)
        artist = html.unescape(re.sub(r"<[^>]+>", "", metadata.get("Artist", {}).get("value", ""))).strip()
        sources.append({"id": image_id, "file_path": relative, "title": title, "author": artist,
            "source": info["descriptionurl"], "download_url": url, "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "sha256": hashlib.sha256(payload).hexdigest()})
        catalog.append({"id": image_id, "file_path": relative, "categories": [], "description": ""})
        print(f"{image_id}: {title} ({len(payload)} bytes)", flush=True)
    (ROOT / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # No human labels are given to the model-generated semantic index.
    (ROOT / "catalog.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in catalog), encoding="utf-8")


if __name__ == "__main__":
    main()
