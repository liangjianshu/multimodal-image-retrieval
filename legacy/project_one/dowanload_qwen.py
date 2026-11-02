# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
from huggingface_hub import snapshot_download
import os

# 模型保存路径（和你代码里的model_path一致）
save_dir = '/path/to/workspace/multimodel_project/mode-Qwen2.5-VL-7B-Instruct'
os.makedirs(save_dir, exist_ok=True)

# 下载完整模型（含14GB权重，断点续传）
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir=save_dir,
    resume_download=True,  # 断点续传，不怕中断
    local_dir_use_symlinks=False,  # 禁用软链接，避免文件缺失
    max_workers=4  # 多线程下载，提速
)

print(f'✅ 模型已完整下载至：{save_dir}')
print(f'✅ 检查文件：ls -lh {save_dir}/*.safetensors')
