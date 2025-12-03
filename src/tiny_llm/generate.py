import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from tiny_llm.kv_cache import TinyKvFullCache
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable
import time


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
    
    start_time = time.time()
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
    end_time = time.time()
    print()
    print(f"[Inference Time] {end_time - start_time:.3f}s")
    return detokenizer.text


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str, max_tokens: int = 100
) -> str:
    def _step(model, y, offset, kv_cache):
        output_logits = model(y, offset, kv_cache)
        logits = output_logits[:, -1, :]
        return mx.argmax(logits, axis=-1) # greedy sampling

    start_time = time.time()
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    offset = 0
    num_generated = 0

    while num_generated < max_tokens:
        next_token = _step(model, tokens, offset, kv_cache)[None, :]
        offset += tokens.shape[1]
        tokens = next_token
        num_generated += 1
        if next_token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(next_token.item())

    detokenizer.finalize()
    end_time = time.time()
    print()
    print(f"[Inference Time] {end_time - start_time:.3f}s")
    return detokenizer.text


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
