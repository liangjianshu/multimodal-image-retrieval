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


def test_batched_collator_masks_each_prompt_and_padding(tmp_path,monkeypatch):
    import sys
    import types
    from mmretrieval.training import SFTCollator
    from PIL import Image
    for name in ['a.png','b.png']:Image.new('RGB',(16,16)).save(tmp_path/name)
    captured=[]
    def vision(messages):
        captured.extend(messages)
        return [m[0]['content'][0]['image'] for m in messages],None
    monkeypatch.setitem(sys.modules,'qwen_vl_utils',types.SimpleNamespace(process_vision_info=vision))
    class Processor:
        tokenizer=types.SimpleNamespace(padding_side='right')
        def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
            prefix=messages[0]['content'][1]['text']+'|'
            return prefix if len(messages)==1 else prefix+messages[1]['content'][0]['text']+'!'
        def __call__(self,text,images,padding,return_tensors):
            assert len(images)==len(text)==2
            tokens=[[ord(c) for c in s] for s in text];width=max(map(len,tokens))
            return {'input_ids':torch.tensor([t+[0]*(width-len(t)) for t in tokens]),
                    'attention_mask':torch.tensor([[1]*len(t)+[0]*(width-len(t)) for t in tokens])}
    result=SFTCollator(Processor(),tmp_path)([
        {'file_path':'a.png','prompt':'short','answer':'A'},
        {'file_path':'b.png','prompt':'longer prompt','answer':'B'}])
    assert len(captured)==2
    assert result['labels'][0,:6].tolist()==[-100]*6
    assert result['labels'][0,6:8].tolist()==[ord('A'),ord('!')]
    assert (result['labels'][0,8:]==-100).all()
    assert result['labels'][1,:14].tolist()==[-100]*14
    assert result['labels'][1,14:].tolist()==[ord('B'),ord('!')]
