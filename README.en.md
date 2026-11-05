# Multimodal Image Retrieval: Two Retrieval Routes, Shared Qwen Intent Parsing

[Route A demo](https://liangjianshu.github.io/multimodal-image-retrieval/) · [Route B demo](https://liangjianshu.github.io/multimodal-image-retrieval/route-b/) · [中文](README.md)

Search a private image library using text, a reference image, or both. **Both routes use Qwen to decide whether retrieval is needed and to parse explicit result counts.** Nonretrieval requests stop before search.

| | Route A: Qwen semantic retrieval | Route B: direct image embeddings |
| --- | --- | --- |
| Gallery | Qwen predicts categories and descriptions; Taiyi embeds the generated text | EmbeddingGemma 2 embeds original pixels |
| Retrieval | Category-vector filtering, followed by description-vector ranking | Direct image-vector ranking; no Qwen category filter |
| Reference input | Qwen describes the target; query vectors can include CLIP image features | Original reference pixels and full text are encoded jointly |
| Training | Qwen category-JSON supervised fine-tuning; descriptions generated at inference | Contrastive query/positive-image/hard-negative-image fine-tuning |
| Public showcase | Validation-selected new Qwen LoRA | **Completed original 2,000-image evaluation, with pretrained control** |

Route A is the default. The original project used Qwen for both category recognition and fine-grained descriptions, not only filtering. An audit found mislabeled negative answers in the preserved historical SFT generator and data. These cannot establish effective hard-negative training, and the exact dataset revision used by the old adapter remains unverified. The supported SFT builder retains correct answers for difficult examples. Route B is an alternative that directly optimizes retrieval vectors. Neither is assumed to outperform the other without a fixed evaluation.

## Completed fixed-gallery evaluation

Eight configurations and 24 checkpoints were evaluated. Checkpoints and thresholds were selected using validation only. On the original 2,000-image gallery, primary-five category mAP is **0.9355 text / 0.8557 image** for the new Embedding model (base: 0.6563 / 0.3827) and **0.8061 / 0.7731** for the new Qwen retrieval pipeline (base: 0.4723 / 0.2687; old LoRA: 0.4124 / 0.2020). These are category-relevance results, not attribute or instance-retrieval accuracy. One seed; no statistical-significance claim.

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

The prior Qwen-category + direct-embedding hybrid is retained as an [experimental record](docs/demo/baseline-hybrid/results.json). Current baselines do not establish general accuracy, fine-tuning gains, production throughput or the superiority of either architecture. Qwen descriptions and generated labels are not human ground truth.

The project uses NumPy indexes, with no production server or vector database. Dataset images, private training data and model weights are not published. Code is [MIT](LICENSE), with separately licensed third-party components; see [notices](THIRD_PARTY_NOTICES.md).
