# 历史源码（非推荐运行入口）

保留原来的 `project_one`、`project_two` 和 `project_three` 结构，方便追溯实验设计。API 凭据已改为环境变量，个人服务器路径已替换为占位路径。请使用仓库根目录 README 中的 `mmretrieval` 命令运行整理版。

这里的文件没有宣称可以不加修改地运行，图片、权重、缓存 JSON 和模型仓库依赖也不包含在快照中。

已知历史问题：

- `project_three/build_dataset.py` 的部分负例把错误类别写进 assistant 答案。它是历史记录，**不要用它生成新的 SFT 标签**；用 `make-sft` 的正确否定问答构造。
- 部分脚本在 import 时加载大型模型，硬编码 GPU 和路径。
- 某些训练范围开关与保存内容不一致，连接层训练后未必被保存；推荐入口显式设置 `modules_to_save`。
- 部分检索阈值探索依赖实验标签；新评测应只在验证集选择阈值。
- 历史 intent 数据生成需要你自己的 `OPENAI_API_KEY`，会调用外部付费 API；推荐运行路径不依赖该接口。

`project_three/Qwen3-VL/` 内为相关微调源码快照和原 Apache-2.0 许可证，不是整个 Qwen 仓库。当前统一训练入口位于 `mmretrieval/training.py`。
