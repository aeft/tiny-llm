import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output_logits = model(y)
        logits = output_logits[:, -1, :]
        return sampler(logits)
    
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    while True:
        next_token = _step(model, tokens)[None, :]
        tokens = mx.concat([tokens, next_token], axis=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(next_token.item())
    detokenizer.finalize()
    print(detokenizer.text)
    return detokenizer.text


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
