# 两条检索路线：共享 Qwen 意图识别

[路线 A：Qwen 类别与描述](https://liangjianshu.github.io/multimodal-image-retrieval/) · [路线 B：微调原图 Embedding](https://liangjianshu.github.io/multimodal-image-retrieval/route-b/) · [路线 B 微调前对照](https://liangjianshu.github.io/multimodal-image-retrieval/route-b-baseline/)

12 张 CC0 图片、10 条固定请求。网页是实际模型输出的交互回放，不是在线推理服务。8 条检索请求与 2 条非检索请求分开展示；所有版本默认没有分数阈值，参考图排除自身。

## 模型职责

| 阶段 | 路线 A | 路线 B |
| --- | --- | --- |
| 是否检索、返回数量 | Qwen2.5-VL-7B＋原 LoRA | 同一 Qwen 的相同意图决定 |
| 图库表示 | Qwen 生成类别与描述，Taiyi 编码文字 | EmbeddingGemma 2 直接编码原图 |
| 筛选与排序 | 目标类别向量粗筛，再按描述向量排序 | 全图库原图向量排序，无类别粗筛 |
| 参考图 | CLIP 图像向量参与融合 | 原始像素与完整文字联合编码 |
| 本次训练 | 新类别监督 LoRA，验证集选择 1 epoch；描述为模型生成 | 新类别对比学习，验证集选择 1 epoch |

路线 A 将 Qwen 提取的目标类别单独用于粗筛，保留完整用户原文进行描述排序。这是对旧流程的明确整理，不等同于逐字复现旧代码中同一个查询向量同时参与两阶段打分的方式。

路线 B 的详细图片说明仅供阅读，不参与检索。两条路线都保留意图识别，知识问答或只描述图片的请求会跳过检索。

## 实际结果

| 请求 | A：类别＋描述 | B：原图微调前 | B：原图微调后 |
| --- | --- | --- | --- |
| white-bicycle | 12, 09, 10 | 09, 12, 10 | 09, 12, 10 |
| red-chair | 01, 03, 02 | 01, 03, 04 | 04, 02, 03 |
| coffee | 06, 07, 08 | 07, 08, 06 | 07, 06, 08 |
| ornate-cup | 08, 06, 05 | 08, 06, 02 | 08, 06, 05 |
| similar-chair | 04, 03, 01 | 03, 01, 04 | 04, 03, 01 |
| similar-cup | 05, 07, 08 | 07, 08, 05 | 07, 08, 05 |
| bicycle-in-grass | 12, 10, 11 | 10, 12, 11 | 10, 12, 11 |
| no-retrieval-qa | 不检索 | 不检索 | 不检索 |
| no-retrieval-description | 不检索 | 不检索 | 不检索 |
| explicit-two-cups | 05, 06 | 07, 08 | 06, 07 |

所有排名和分数都来自模型实跑，未手工改序。明确“两张杯子”的请求在各版本返回两张；两条非检索请求均返回零张。

## 已知失败与实验边界

- 路线 B 的“花纹杯”前三名现在均为杯子；路线 A 的“白色自行车”“红色椅子”仍有属性排序错误。类别训练不等同于颜色、材质或实例监督，真实失败结果保留。
- 路线 A 的粗筛可能漏掉可接受的近似类别；描述会丢失视觉细节。两个不同编码器的分数不能直接比较大小。
- 12 张图片仅做定性演示。正式数值来自恢复的原始 2,000 张固定图库，见 [正式实验报告](RESTORED_STUDY.md)。固定基准使用类别相关性，不能代替自由属性查询的准确率。
- 历史 Qwen 数据生成脚本存在把负例错误类别写成 SFT 正确答案的问题；原适配器所用数据的精确版本尚未核实。不能把类别筛选的收益直接归因于那批负例训练。
- 微调版复用已经生成并校验一致的 Qwen 意图，避免把路由随机变化混入编码器对比。计时字段区分缓存意图与原始意图耗时，不将其当成完整在线延迟基准。

## 记录与权重

- [路线 A 原始结果](results.json)
- [路线 B 微调前](route-b-baseline/results.json)
- [路线 B 微调后](route-b/results.json)
- [本轮共享 Qwen 意图](demo/study-20261105/shared-intents.json)
- [本轮选型、数据与模型哈希](demo/study-20261105/provenance.json)
- [新类别适配器逐张量校验](demo/study-20261105/category-adapter-verification.json)
- [原 Qwen 适配器逐张量校验](demo/adapter-verification.json)
- [训练配置、梯度日志与模型哈希](experiments/embedding-category-pilot-v1/training_report.json)

原 Qwen 适配器为语言注意力 LoRA，rank 64、alpha 128；没有单独保存 merger，不能将加载产物称为完整 MLP＋LLM 微调模型。Embedding 本轮训练更新文字与视觉模块全部 438,760,448 个参数；权重保存在作者服务器，不随仓库分发。

## 复跑

通过 [README](../README.md) 选择 A/B 配置。训练见 [Embedding 指南](EMBEDDING_TRAINING.md)。`examples/showcase/run_routes.py` 生成共享意图并运行 A/B；`run_embedding_checkpoint.py` 校验图库与描述缓存后，将相同意图交给新编码器，记录真实结果。`build_page.py` 只渲染结果，不修改排名。

[旧混合版归档](DEMO_HYBRID.md)保留 Qwen 类别筛选＋未微调原图编码的历史流程。

2026-11-05 本轮正式复跑使用 `examples/showcase/run_restored_winners.py`，读取测试前冻结的 `selection.json`。之前的 64 步试训保留在独立报告及 Git 历史中。
