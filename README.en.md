# Multimodal Retrieval Lab: Qwen LoRA × Contrastive Embedding Training

[Project overview](https://liangjianshu.github.io/multimodal-image-retrieval/overview/) · [Route A demo](https://liangjianshu.github.io/multimodal-image-retrieval/) · [Route B demo](https://liangjianshu.github.io/multimodal-image-retrieval/route-b/) · [Learning path (中文)](docs/LEARNING_PATH.md) · [中文](README.md)

Search a private image library using text, a reference image, or both. **Both routes use Qwen to decide whether retrieval is needed and to parse explicit result counts.** Nonretrieval requests stop before search.

Learn the path from model inference to **intent recognition, prompt engineering, Qwen-VL LoRA / SFT, EmbeddingGemma 2, multi-positive InfoNCE, and controlled retrieval evaluation**. The repository includes recorded demos, custom-data instructions, training code and fixed-gallery results. Author-trained weights are not distributed; public configurations use base models and you can train your own.

[![Project overview and learning entry points](docs/overview/preview.png)](https://liangjianshu.github.io/multimodal-image-retrieval/overview/)

| Learn / build | Implementation |
|---|---|
| Multimodal retrieval | Text, reference-image and image-text queries |
| Intent recognition and routing | Retrieval decisions, target and constraint extraction, explicit result counts |
| Prompt engineering | Task-specific intent/category/description prompts, structured JSON, examples, negation and condition preservation, object-count vs. result-count disambiguation |
| Qwen-VL adaptation | Correct category-JSON SFT with PEFT LoRA; descriptions generated at inference |
| Direct image embeddings | EmbeddingGemma 2, 768-dimensional normalized vectors and cosine ranking |
| Contrastive learning | Text-image / image-image training with multi-positive class InfoNCE |
| Reliable evaluation | Validation-selected checkpoints and thresholds, F1 / mAP, format and category controls |
| Multimodal RAG / Agent extension | A retrieval component that can supply images and metadata; answer generation, citations and full agent orchestration are not implemented |

Start with the [demo](https://liangjianshu.github.io/multimodal-image-retrieval/route-b/), then [use your own images](docs/CUSTOM_DATA.md), [fine-tune Qwen](docs/TRAINING.md) or [train embeddings](docs/EMBEDDING_TRAINING.md), and inspect the [evaluation](docs/RESTORED_STUDY.md).

| | Route A: Qwen semantic retrieval | Route B: direct image embeddings |
| --- | --- | --- |
| Gallery | Qwen predicts categories and descriptions; Taiyi embeds the generated text | EmbeddingGemma 2 embeds original pixels |
| Retrieval | Category-vector filtering, followed by description-vector ranking | Direct image-vector ranking; no Qwen category filter |
| Reference input | Qwen describes the target; query vectors can include CLIP image features | Original reference pixels and full text are encoded jointly |
| Training | Qwen category-JSON supervised fine-tuning; descriptions generated at inference | Contrastive query/positive-image/hard-negative-image fine-tuning |
| Public showcase | Validation-selected new Qwen LoRA | **Completed original 2,000-image evaluation, with pretrained control** |

Route A is the default. The original project used Qwen for both category recognition and fine-grained descriptions, not only filtering. An audit found mislabeled negative answers in the preserved historical SFT generator and data. These cannot establish effective hard-negative training, and the exact dataset revision used by the old adapter remains unverified. The supported SFT builder retains correct answers for difficult examples. Route B is an alternative that directly optimizes retrieval vectors. Neither is assumed to outperform the other without a fixed evaluation.

## Completed fixed-gallery evaluation

Eight configurations and 24 checkpoints were evaluated with 4,500 training, 1,000 validation and 2,000 test images. Checkpoints and thresholds were selected using validation only. Both routes now report primary-five **retrieval F1** below. These are category-relevance results, not attribute or instance-retrieval accuracy. One seed; no statistical-significance claim.

| Version (original frozen protocol) | Text F1 | Image F1 |
|---|---:|---:|
| Qwen base pipeline | 60.62% | 35.64% |
| New Qwen LoRA pipeline | **86.76%** | **81.86%** |
| Old Qwen LoRA pipeline | 51.59% | 28.92% |
| Embedding base | 61.68% | 31.85% |
| Fine-tuned Embedding | **88.84%** | **83.40%** |

**Qwen attribution control:** prompts, category aliases and generation settings were already identical. However, strict JSON parsing rejects some otherwise recoverable category outputs. Applying the same structural normalization to both models and recalibrating thresholds on validation only gives **76.07% / 63.90%** for the base and **86.76% / 81.86%** for the fine-tuned pipeline: **+10.69 / +17.96 percentage points**. This post-hoc control is separate from the frozen test; it does not establish pure visual-capability gains. Fixed-filter controls localize most of this task's benefit to category filtering. See [controls and reproduction](docs/QWEN_CONTROLS.md).

Embedding F1 gains in the frozen protocol are **+27.16 / +51.55 percentage points**, with no Qwen category-output parsing. Both routes also retain mAP in the full report.

**Training:** Qwen uses correct category-JSON SFT and language-attention LoRA (`rank=64, alpha=128`), with a separate visual-merger joint-training comparison. Validation selected `lr=1e-5, epoch=1, LLM LoRA`. Embedding uses full-parameter, multi-positive class InfoNCE for text-image and image-image queries; same-class in-batch candidates are positives, not false negatives. Validation selected `lr=2e-6, temperature=0.1, epoch=1`.

**Earlier intent measurements:** text / image-text accuracy **95% / 91%**, retained from the author's earlier intent experiment. Intent was not re-evaluated in this fixed-category retrieval study. Historical retrieval F1 85% / 81% and +6 / +7-point gains remain separate historical records.

[Full report and CSV](https://liangjianshu.github.io/multimodal-image-retrieval/study/) · [Protocol and recovery](docs/RESTORED_STUDY.md). The earlier 297-image pilot remains a separate historical experiment. Current demos use the new validation-selected models and shared original-Qwen intent decisions. Weights are not distributed.

## Run either route

Python 3.10+. Install the matching PyTorch build first, then `pip install -e '.[hybrid]'` in an independent environment. Both models are loaded locally.

```bash
python -m mmretrieval prepare --image-root data/images --output data/catalog.jsonl
python -m mmretrieval --config configs/route-a-qwen-semantic.json index \
  --manifest data/catalog.jsonl --image-root data/images --output artifacts/index-a
python -m mmretrieval --config configs/route-a-qwen-semantic.json search \
  --index artifacts/index-a --image-root data/images --text '找两张红色杯子图片'
```

For Route B, use `configs/route-b-embedding.json` and a separate `artifacts/index-b`. Add `--image queries/reference.jpg` for reference-image queries. Qwen intent parsing is automatic in both routes; no `--intent` flag is required. Public configurations use the Qwen base; set `adapter` to use a compatible LoRA. The author's adapter is not distributed.

A real 64-step full-parameter pilot is documented in the [experiment report](docs/EMBEDDING_EXPERIMENT.md). Evaluation uses the same 297-image held-out gallery before and after training. This measures category retrieval only; fine-grained attribute gains and superiority over the old Qwen model are not established. Public configs still load the pretrained base; private weights are not distributed.

## Train Route B

```bash
python -m mmretrieval --config configs/route-b-embedding.json train-embedding \
  --train data/triplets/train.jsonl --val data/triplets/val.jsonl \
  --image-root data/images --output checkpoints/embedding-run-01 \
  --epochs 1 --learning-rate 0.000002 --grad-accum 4
```

Each JSONL row supplies an `id`, query `text` and/or reference `image`, plus `positive` and `negative` image paths. The trainer checks image-content overlap between splits. It saves weights, training/validation records, hashes and an inference configuration. Rebuild gallery embeddings with the new checkpoint. See [training guide](docs/EMBEDDING_TRAINING.md). Explicit-pair validation accuracy is not full-gallery recall.

Historical Qwen training uses `.[models]` in a separate environment. See [Qwen training](docs/TRAINING.md) and [historical experiments](docs/EXPERIMENTS.md).

## Evidence and limits

The 12-image, 10-request showcase replays actual server outputs; it is not a live inference service. Both routes share the same recorded Qwen decisions. Retrieval and nonretrieval examples are displayed separately. There is no default score cutoff, and reference images are excluded from their own results. Raw outputs, limitations and reproduction are in [DEMO.md](docs/DEMO.md).

The prior Qwen-category + direct-embedding hybrid is retained as an [experimental record](docs/demo/baseline-hybrid/results.json). The fixed-gallery study establishes gains only under its documented category-relevance protocol, not general accuracy, production throughput or universal architectural superiority. Qwen descriptions and generated labels are not human ground truth.

The project uses NumPy indexes, with no production server or vector database. Dataset images, private training data and model weights are not published. Code is [MIT](LICENSE), with separately licensed third-party components; see [notices](THIRD_PARTY_NOTICES.md).
