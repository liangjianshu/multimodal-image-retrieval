# 整理版验证记录（2026-11-02）

本地环境：Python 3.13.5、PyTorch 2.9.0、Torchvision 0.24.0、Transformers 4.57.1、PEFT 0.17.1、Accelerate 1.11.0、qwen-vl-utils 0.0.14。

- `pytest -q`：16 项通过。覆盖自有图片目录导入、路径/ID 校验、重复图片隔离、难负例的正确答案、两阶段排序、索引读写、参考图排除、检索指标和 CLI 数据流程。
- 微型随机 Qwen：三种微调范围检查通过；多模态 forward/backward 和优化器更新通过；语言 LoRA 与 merger 都发生更新；merger 保存为适配器后重新加载保持一致。
- 官方 Qwen2.5-VL-3B processor/tokenizer 的单样本集成检查通过：47 个输入 token、6 个监督 token、图像张量形状 `[16,1176]`，prompt 标签正确遮罩。仅下载处理器/分词器，没有加载 3B 预训练权重。
- CLI 合成 demo 运行成功；六张自生成几何图片，索引与训练划分落盘成功。预设向量的 F1 不作为模型效果指标。
- 待提交文件的凭据、私人绝对路径与 Python 语法检查通过；Git whitespace 检查通过。

尚未验证：真实 3B/7B 模型的完整图片描述、检索效果、训练收敛、GPU 内存峰值与吞吐，以及原历史 6/7 个百分点收益。新硬件或新数据使用前应先小样本验证。

GitHub Actions 配置了 CPU 自动测试，每次提交会重新运行测试和合成 demo；实际运行状态以 Actions 页面为准。
