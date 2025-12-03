import mlx.core as mx
from mlx_lm import load
from tiny_llm.models import dispatch_model, shortcut_name_to_full_name
from tiny_llm.generate import simple_generate_with_kv_cache
from tiny_llm.utils import prepare_prompt
import pytest

@pytest.mark.parametrize("model_name", ["qwen2-0.5b", "qwen2-1.5b"])
@pytest.mark.parametrize("prompt", ["Give me a short introduction to large language model."])
def test_model_with_flash_attention(model_name, prompt):
    model_name = shortcut_name_to_full_name(model_name)
    model, tokenizer = load(model_name)
    prompt = prepare_prompt(tokenizer, prompt)
    with mx.stream(mx.gpu):
        model_std = dispatch_model(model_name, model, week=2, enable_flash_attn=False)
        result_std = simple_generate_with_kv_cache(model_std, tokenizer, prompt, max_tokens=50)
        del model_std

        model_flash = dispatch_model(model_name, model, week=2, enable_flash_attn=True)
        result_flash = simple_generate_with_kv_cache(model_flash, tokenizer, prompt, max_tokens=50)
        del model_flash

    print(f"Result std: {result_std}")
    print()
    print(f"Result flash: {result_flash}")

    assert result_std == result_flash
