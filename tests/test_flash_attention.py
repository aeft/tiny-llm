import mlx.core as mx
import tiny_llm
import tiny_llm_ref
from tests.utils import assert_allclose

def test_flash_attention_gqa_cpu():
    mx.random.seed(42)
    B, H_q, H_kv, L, D = 2, 8, 2, 64, 32
    
    q = mx.random.normal((B, H_q, L, D))
    k = mx.random.normal((B, H_kv, L, D))
    v = mx.random.normal((B, H_kv, L, D))
    
    with mx.stream(mx.cpu):
        ref = tiny_llm_ref.flash_attention(q, k, v, scale=1.0)
        result = tiny_llm.flash_attention(q, k, v, scale=1.0)
        
        assert_allclose(result, ref, precision=mx.float16, rtol=1e-4, atol=1e-4)

def test_flash_attention_gqa_large_cpu():
    mx.random.seed(42)
    B, H_q, H_kv, L, S, D = 4, 8, 2, 64, 128, 128
    
    q = mx.random.normal((B, H_q, L, D))
    k = mx.random.normal((B, H_kv, S, D))
    v = mx.random.normal((B, H_kv, S, D))
    
    with mx.stream(mx.cpu):
        ref = tiny_llm_ref.flash_attention(q, k, v, scale=1.0)
        result = tiny_llm.flash_attention(q, k, v, scale=1.0)
        
        assert_allclose(result, ref, precision=mx.float16, rtol=1e-4, atol=1e-4)

def test_flash_attention_gqa_gpu():
    mx.random.seed(42)
    B, H_q, H_kv, L, D = 2, 8, 2, 64, 32
    
    q = mx.random.normal((B, H_q, L, D))
    k = mx.random.normal((B, H_kv, L, D))
    v = mx.random.normal((B, H_kv, L, D))
    
    with mx.stream(mx.gpu):
        ref = tiny_llm_ref.flash_attention(q, k, v, scale=1.0)
        result = tiny_llm.flash_attention(q, k, v, scale=1.0)
        
        assert_allclose(result, ref, precision=mx.float16, rtol=1e-4, atol=1e-4)

def test_flash_attention_gqa_large_gpu():
    mx.random.seed(42)
    B, H_q, H_kv, L, S, D = 4, 8, 2, 64, 128, 128
    
    q = mx.random.normal((B, H_q, L, D))
    k = mx.random.normal((B, H_kv, S, D))
    v = mx.random.normal((B, H_kv, S, D))
    
    with mx.stream(mx.gpu):
        ref = tiny_llm_ref.flash_attention(q, k, v, scale=1.0)
        result = tiny_llm.flash_attention(q, k, v, scale=1.0)
        
        assert_allclose(result, ref, precision=mx.float16, rtol=1e-4, atol=1e-4)