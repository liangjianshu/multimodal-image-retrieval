# 参与贡献 / Contributing

欢迎反馈自己的图片库接入经验、修复问题或改进文档。首次参与可以从补充使用示例、完善错误提示和改进数据校验开始。

## 报告问题

请通过 Issue 模板提供：系统/Python/依赖版本、完整命令、最小复现步骤，以及预期与实际行为。贴日志前移除 API 密钥、访问令牌和私人图片路径；尽量使用可公开的最小样例。

## 修改代码

1. Fork 仓库并创建分支。
2. 使用隔离环境安装 `pip install -e '.[dev]'`；涉及模型时安装 `.[models,dev]`。
3. 针对行为变化增加必要测试，运行 `pytest -q` 和 `python -m mmretrieval demo`。
4. 提交 PR，说明问题、改动、验证范围和已知限制。

请把改动集中在一件事上。新增模型依赖应保持延迟加载，避免让数据准备和合成 demo 被迫下载权重。不要提交模型权重、第三方数据集、个人配置或密钥。

## 效果报告

分享检索收益时，请同时提供数据来源、划分方式、参考图排除规则、相关性标注、模型/适配器版本和阈值。区分合成测试、随机微型模型检查与真实模型评测。历史实验分数不能自动作为新功能的效果证明。

## 贡献许可

项目贡献采用仓库 MIT License，另有许可证声明的第三方文件除外。修改第三方文件时保留其原始版权与许可证说明。

English: please include a minimal reproduction for bug reports and describe validation in pull requests. Keep changes focused, avoid committing private data or weights, and retain third-party license notices.
