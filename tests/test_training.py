"""Optional tiny random Qwen tests: no pretrained weights, no reported task scores."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

from mmretrieval.training import configure_training


def tiny_model():
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "patch_size": 14, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "out_hidden_size": 32,
                       "fullatt_block_indexes": [0]},
        image_token_id=6, video_token_id=8, vision_start_token_id=5, vision_end_token_id=7)
    return Qwen2_5_VLForConditionalGeneration(config)


@pytest.mark.parametrize("mode", ["llm-lora", "mlp-llm", "mlp-vision"])
def test_trainable_scope_matches_mode(mode):
    model, trainable = configure_training(tiny_model(), mode, rank=2, alpha=4)
    if mode == "llm-lora":
        assert all("lora_" in n and "visual" not in n for n in trainable)
    elif mode == "mlp-llm":
        assert any("merger" in n for n in trainable)
        assert any("lora_" in n for n in trainable)
        assert all("merger" in n or "lora_" in n for n in trainable)
    else:
        assert all("visual." in n for n in trainable)
        assert any("merger" in n for n in trainable)


def test_mlp_weights_survive_adapter_save_reload(tmp_path):
    model, _ = configure_training(tiny_model(), "mlp-llm", rank=2, alpha=4)
    merger = [(n, p) for n, p in model.named_parameters() if "merger" in n and p.requires_grad]
    assert merger
    with torch.no_grad():
        for _, parameter in merger:
            parameter.fill_(0.123)
    model.save_pretrained(tmp_path / "adapter")
    reloaded = PeftModel.from_pretrained(tiny_model(), tmp_path / "adapter")
    weights = [p for n, p in reloaded.named_parameters() if "merger" in n and "modules_to_save.default" in n]
    assert weights
    assert all(torch.allclose(p, torch.full_like(p, 0.123)) for p in weights)


def test_tiny_multimodal_training_step_updates_merger_and_lora():
    torch.manual_seed(42)
    model, trainable = configure_training(tiny_model(), "mlp-llm", rank=2, alpha=4)
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    input_ids = torch.tensor([[10, 5, 6, 6, 6, 6, 7, 11, 12]])
    labels = input_ids.clone()
    labels[:, :7] = -100
    result = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels,
                   pixel_values=torch.randn(16, 3 * 2 * 14 * 14), image_grid_thw=torch.tensor([[1, 4, 4]]))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if n in before and not torch.equal(before[n], p)]
    assert any("merger" in n for n in changed)
    assert any("lora_" in n for n in changed)
