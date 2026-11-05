"""Preview recorded retrieval results using only Python's standard library."""

import argparse
import hashlib
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_assets():
    sources = json.loads((ROOT / "examples/showcase/sources.json").read_text())
    for item in sources:
        path = ROOT / "examples/showcase" / item["file_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Image checksum mismatch: {path}")
        published = ROOT / "docs/route-b" / item["file_path"]
        if not published.is_file():
            raise FileNotFoundError(published)
    for page in ("overview/index.html", "index.html", "route-b/index.html",
                 "route-b-baseline/index.html", "study/index.html"):
        if not (ROOT / "docs" / page).is_file():
            raise FileNotFoundError(page)
    print(f"OK: {len(sources)} source images verified; preview assets present.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check assets and exit")
    parser.add_argument("--port", type=int, default=8000, help="Local port (0 chooses a free port)")
    args = parser.parse_args()
    try:
        check_assets()
        if args.check:
            return
        handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "docs"))
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"Open: http://127.0.0.1:{server.server_port}/overview/", flush=True)
            print("Recorded results only; no model inference or downloads. Ctrl+C to stop.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nPreview stopped.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Quickstart failed: {error}\nTry --port 8001 if the port is occupied.\n")


if __name__ == "__main__":
    main()
