# 来源与整理范围

项目由多模态检索实践及后续个人实验整理。原工程包含三个目录：

| 原目录 | 内容 | 整理后入口 |
| --- | --- | --- |
| `project_one` | 图片数据集、CLIP 检索、Qwen 描述建库、LoRA 训练 | `prepare` / `describe` / `index` / `search` / `train` |
| `project_two` | 意图数据生成、文本/图文意图识别、意图评测 | `search --intent`；历史数据生成和评测仍在 legacy |
| `project_three` | 联合检索、训练数据生成、Qwen 微调框架 | `search` / `make-sft` / `train` |

`legacy/` 是源码清理快照：移除内嵌 API 凭据、替换私人绝对路径，保留历史结构供追溯。完整模型、图片、生成数据、私有配置、缓存和 Git 历史不在仓库中。原始服务器目录没有被修改。

`mmretrieval/`、测试、文档与配置属于 2026-11-02 整理过程中新增的通用实现，复用了原项目的检索设计并修复可复用性问题。它们不应被描述成原实验时已经具备的工程能力。历史脚本与推荐入口的行为差异见 `EXPERIMENTS.md`。

历史 Qwen 微调文件来自 [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)，服务器仓库记录 HEAD 为 `9658872`；本地文件可能存在后续调整，此提交号不代表每个快照文件与上游完全一致。保留 `legacy/project_three/Qwen3-VL/LICENSE`（Apache-2.0）。

2026-11-03，项目维护者确认有权公开所提交代码，并授权发布。项目代码采用仓库根目录的 MIT License；另有声明的第三方内容继续适用原许可证，尤其是上述 Qwen 源码的 Apache-2.0 许可。此记录用于说明发布依据，不代表第三方对本项目的背书。
