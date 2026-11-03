# Multimodal Image Retrieval

**Search your own image library with Chinese text, a reference image, or both.**

[Online demo](https://liangjianshu.github.io/multimodal-image-retrieval/) · [中文文档](README.md) · [Issues](https://github.com/liangjianshu/multimodal-image-retrieval/issues) · [Contributing](CONTRIBUTING.md)

A learning and prototyping project for visual product search, image assets and private image libraries. It combines Taiyi-CLIP embeddings, Qwen2.5-VL semantic indexing, category filtering and description-vector ranking, with optional domain adaptation using LoRA / SFT.

It also demonstrates components relevant to agent applications: intent parsing, retrieval pipelines and model invocation. It is not a full agent orchestration framework or a production-scale search service. References to Pailitao describe a use case; this project is independent of Alibaba and Taobao.

## What you can explore

- **Text-to-image:** retrieve images using a Chinese description.
- **Image-to-image:** find visually related products or assets from a reference image.
- **Combined queries:** fuse a text embedding with a reference image embedding.
- **Bring your own data:** use class folders or a JSONL manifest with image paths and metadata.
- **Optional adaptation:** compare language LoRA, merger + language LoRA, and full visual-tower tuning.
- **Evaluation:** compute per-query precision, recall and F1 using explicit relevance labels.

The index is an in-memory NumPy matrix. External vector databases, a live inference web service and production performance benchmarks are not included.

## Real-model showcase

![Real text-to-image retrieval on a 12-image CC0 gallery](docs/demo/text-search.png)

Seven recorded result sets use **Qwen2.5-VL-7B with the author’s original LoRA adapter**. The model generates gallery metadata and parses image-query targets; Taiyi-CLIP fuses the target text and reference image before category filtering and description ranking. No manual labels, synthetic vectors or manual reranking are used. Reference images are excluded from their own results.

Try the [online demo](https://liangjianshu.github.io/multimodal-image-retrieval/), or open `docs/demo/index.html` offline. Images are embedded, so the HTML works as a standalone file; its initial results also render without JavaScript. This is a replay, not browser-side model inference. See the [reproduction guide](docs/DEMO.md), [raw outputs](docs/demo/results.json) and [CC0 image credits](examples/showcase/SOURCES.md).

This is a small-gallery qualitative replay, not a benchmark or a reproduction of historical F1 gains. The original adapter is required to reproduce the main showcase and is not distributed here. The earlier 3B/no-adapter outputs remain in [baseline records](docs/demo/baseline-3b/results.json). Model size, prompts and query processing also changed, so differences cannot be attributed to LoRA alone.

## Quick start without model downloads

```bash
git clone https://github.com/liangjianshu/multimodal-image-retrieval.git
cd multimodal-image-retrieval
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m mmretrieval demo --output artifacts/demo
```

The demo uses generated geometric images and **predefined synthetic embeddings** to check the data flow. Its scores are not model-quality measurements.

## Use real images

Install the model dependencies with `pip install -e '.[models]'`. Linux with a CUDA GPU is recommended for Qwen inference and training. Install the appropriate PyTorch build for your hardware before installing the extras.

Prepare a manifest, with image paths relative to `data/images`:

```json
{"id":"chair-001","file_path":"chair/001.jpg","categories":["椅子"],"description":"一把红色木质椅子"}
```

Or generate one from class folders:

```bash
python -m mmretrieval prepare --image-root data/images \
  --category-map examples/category_map.json --output data/catalog.jsonl
python -m mmretrieval validate --manifest data/catalog.jsonl --image-root data/images
python -m mmretrieval --config configs/default.json describe \
  --manifest data/catalog.jsonl --image-root data/images --output data/catalog.described.jsonl
python -m mmretrieval --config configs/default.json index \
  --manifest data/catalog.described.jsonl --image-root data/images --output artifacts/index
python -m mmretrieval search --index artifacts/index --image-root data/images \
  --text '红色木质椅子' --top-k 5
```

The default text encoder targets Chinese queries. This English overview does not imply that the default model configuration has been evaluated for English retrieval.

## Documentation

- [Full usage guide (Chinese)](README.md)
- [Custom datasets, manifests and relevance labels (Chinese)](docs/CUSTOM_DATA.md)
- [Training modes and loading saved adapters (Chinese)](docs/TRAINING.md)
- [Historical results and reproducibility limits (Chinese)](docs/EXPERIMENTS.md)
- [Validation performed (Chinese)](docs/VERIFICATION.md)

Historical F1 improvements are author-reported results and have not been reproduced with this refactored release. Tests validate program behavior, including a small randomly initialized Qwen training step and adapter round-trip; they do not establish full-size model quality.

## Feedback and license

Try the pipeline on a small collection first. Report setup issues, use cases or reproducible errors through [Issues](https://github.com/liangjianshu/multimodal-image-retrieval/issues/new/choose). If it helps you learn or build, consider starring the project.

[MIT License](LICENSE), except for separately licensed third-party files. The historical Qwen directory retains Apache-2.0; the bundled showcase photos are CC0. See [third-party notices](THIRD_PARTY_NOTICES.md). Model weights and the original COCO/ImageNet datasets are not included.
