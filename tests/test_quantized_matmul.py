import pytest
import mlx.core as mx
from extensions import tiny_llm_ext
from .utils import assert_allclose


@pytest.fixture
def random_inputs():
    """Random test case: 64x128 @ 128x128.T"""
    mx.random.seed(42)
    a = mx.random.normal(shape=(64, 128), dtype=mx.float16)
    weight = mx.random.normal(shape=(128, 128), dtype=mx.float16)
    w_q, scales, biases = mx.quantize(weight, group_size=64, bits=4)
    return a, w_q, scales, biases


def test_quantized_matmul_cpu(random_inputs):
    """Test CPU implementation"""
    a, w_q, scales, biases = random_inputs
    
    result = tiny_llm_ext.quantized_matmul(
        scales, biases, 64, 4, a, w_q, transpose_b=True, stream=mx.cpu
    )
    mx.eval(result)
    
    expected = mx.quantized_matmul(
        a, w_q, scales, biases, transpose=True, group_size=64, bits=4
    )
    mx.eval(expected)
    
    assert_allclose(result, expected, precision=mx.float16, rtol=1e-4, atol=1e-4)
    print(f"✅ CPU test passed: {result.shape}")


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_quantized_matmul_gpu(random_inputs):
    """Test GPU implementation"""
    a, w_q, scales, biases = random_inputs
    
    with mx.stream(mx.gpu):
        a = mx.array(a)
        w_q = mx.array(w_q)
        scales = mx.array(scales)
        biases = mx.array(biases)
        
        result = tiny_llm_ext.quantized_matmul(
            scales, biases, 64, 4, a, w_q, transpose_b=True, stream=mx.gpu
        )
        mx.eval(result)
        
        expected = mx.quantized_matmul(
            a, w_q, scales, biases, transpose=True, group_size=64, bits=4
        )
        mx.eval(expected)
        
        assert_allclose(result, expected, precision=mx.float16, rtol=1e-4, atol=1e-4)
        print(f"✅ GPU test passed: {result.shape}")