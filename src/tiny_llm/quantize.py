import mlx.core as mx
from typing import Any
from extensions import tiny_llm_ext
from typing import Literal


def dequantize_linear(mx_layer: Any) -> mx.array:
    assert mx_layer.group_size == 64
    assert mx_layer.bits == 4
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    # Flatten leading dimensions
    *N, D = a.shape
    a = a.reshape(-1, D)
    
    # Ensure contiguous memory
    a = mx.contiguous(a)
    b = mx.contiguous(b)
    
    result = tiny_llm_ext.quantized_matmul(
        scales, biases, group_size, bits, a, b, transpose_b
    )
    
    # Restore original batch dimensions
    return result.reshape(*N, -1)

_QUANTIZED_LINEAR_MODE: Literal["custom", "dequantize", "mlx"] = "dequantize"

def set_quantized_linear_mode(mode: Literal["custom", "dequantize", "mlx"]):
    """
    Set the implementation mode for quantized_linear
    
    Args:
        mode: 
            - "custom": Use custom C++ implementation
            - "dequantize": Use dequantize + linear (default)
            - "mlx": Use MLX built-in quantized_matmul
    """
    global _QUANTIZED_LINEAR_MODE
    assert mode in ["custom", "dequantize", "mlx"], f"Invalid mode: {mode}"
    _QUANTIZED_LINEAR_MODE = mode
    print(f"quantized_linear mode set to: {mode}")

def get_quantized_linear_mode() -> str:
    """Get current quantized_linear mode"""
    return _QUANTIZED_LINEAR_MODE

def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:

    if _QUANTIZED_LINEAR_MODE == "custom":
        result = quantized_matmul(
            w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
        )
    elif _QUANTIZED_LINEAR_MODE == "dequantize":
        result = x @ dequantize_linear(w).T
    elif _QUANTIZED_LINEAR_MODE == "mlx":
        result = mx.quantized_matmul(x, w.weight, w.scales, w.biases, True, w.group_size, w.bits, stream=mx.gpu)
    else:
        raise ValueError(f"Invalid mode: {_QUANTIZED_LINEAR_MODE}")

    if bias is not None:
        result = result + bias
    return result
