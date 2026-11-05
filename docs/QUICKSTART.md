# 新手快速开始：先看结果，再跑模型

## 1. 无需 GPU：本地查看真实实验的记录

准备 Git 和 Python 3.10+，在终端执行：

```bash
git clone https://github.com/liangjianshu/multimodal-image-retrieval.git
cd multimodal-image-retrieval
python3 examples/quickstart.py
```

Windows 可将 `python3` 换为 `python`。打开终端显示的 `http://127.0.0.1:8000/overview/`，即可查看项目介绍、切换检索案例，以及阅读实验对照。按 `Ctrl+C` 停止。端口被占用时使用 `--port 8001`。

这个入口只使用 Python 标准库，不需要安装 PyTorch、下载模型或填写 API Key。它在本机提供 `docs/` 下的静态页面，**回放已经记录的模型输出，不接受自由输入，也不执行实时推理**。联网链接会进入 GitHub 或在线文档。

仓库已包含 12 张 CC0 示例图片，出处见 [SOURCES.md](../examples/showcase/SOURCES.md)。也可以只检查资源完整性：

```bash
python3 examples/quickstart.py --check
```

没有本地 Python？直接打开[在线项目主页](https://liangjianshu.github.io/multimodal-image-retrieval/overview/)。

## 2. 有模型运行环境：用这 12 张图实际检索

这一步会下载、加载公开模型，需要网络、磁盘和足够的运行内存；与上面的静态预览不同。先建立独立环境，并按硬件安装兼容的 PyTorch，再安装项目依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hybrid]'
python -m mmretrieval validate --manifest examples/showcase/catalog.jsonl --image-root examples/showcase
```

Windows PowerShell 的激活命令为 `.venv\Scripts\Activate.ps1`。首次模型下载若要求授权，请按模型提供方的要求完成。资源不足时先使用静态演示，不必为了看结果安装模型环境。

以路线 B 为例，建库一次后可更换查询：

```bash
python -m mmretrieval --config configs/route-b-embedding.json index \
  --manifest examples/showcase/catalog.jsonl \
  --image-root examples/showcase --output artifacts/quickstart-b

python -m mmretrieval --config configs/route-b-embedding.json search \
  --index artifacts/quickstart-b --image-root examples/showcase \
  --text '找两张杯子的图片'
```

以上多行命令使用 Bash 换行语法；PowerShell 可合并成一行执行。建库由 Embedding 编码原图；查询由 Qwen 判断检索意图，再进入向量检索。`--config` 必须放在 `index` / `search` 前。

**公开配置加载基座模型。** 作者微调权重尚未随仓库分发，所以这一步的输出不保证与微调演示一致；训练自己的版本后，需要用对应配置重新建库。这 12 张图片用于体验流程，不是计算论文式指标的 2,000 张正式测试集。

注意使用 `examples/showcase/catalog.jsonl`；`examples/catalog.jsonl` 是另一份格式示例，其引用图片不随仓库提供。

## 3. 换数据与学习训练

- [换成自己的图库](CUSTOM_DATA.md)：准备图片与 JSONL，再建索引。
- [理解意图识别和 Prompt](INTENT_AND_EMBEDDINGS.md)：保留条件，区分普通问答与检索。
- [Qwen 微调](TRAINING.md)与 [Embedding 对比学习](EMBEDDING_TRAINING.md)：准备监督，训练后验证。
- [实验口径](RESTORED_STUDY.md)与[消融](QWEN_CONTROLS.md)：区分模型、格式和类别筛选的贡献。

如果运行失败，提交 Issue 时请附上系统、Python 版本、执行命令及错误信息。模型运行问题请同时说明硬件、配置与模型版本；不要上传访问令牌或私人图片。
