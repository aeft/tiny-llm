from functools import reduce
import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = 1 / (key.shape[-1] ** 0.5)
    scores = (query @ mx.swapaxes(key, -2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    return softmax(scores, -1) @ value

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        assert self.hidden_size % self.num_heads == 0
        mh_q = linear(query, self.wq)
        assert mh_q.shape[-1] == self.hidden_size
        mh_q = mh_q.reshape(*mh_q.shape[:-1], self.num_heads, mh_q.shape[-1] // self.num_heads).swapaxes(-3, -2)
        mh_k = linear(key, self.wk)
        mh_k = mh_k.reshape(*mh_k.shape[:-1], self.num_heads, mh_k.shape[-1] // self.num_heads).swapaxes(-3, -2)
        mh_v = linear(value, self.wv)
        mh_v = mh_v.reshape(*mh_v.shape[:-1], self.num_heads, mh_v.shape[-1] // self.num_heads).swapaxes(-3, -2)
        mh_o = scaled_dot_product_attention_simple(mh_q, mh_k, mh_v, mask=mask)
        mh_o = mh_o.swapaxes(-3, -2)
        mh_o = mh_o.reshape(*mh_o.shape[:-2], self.hidden_size)
        return linear(mh_o, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    xrange = mx.arange(L).reshape(L, 1)
    yrange = mx.arange(S).reshape(1, S)
    mask = (yrange > (S - L + xrange))
    mask = mx.where(mask, mx.array(-mx.inf), mx.array(0)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    if scale is None:
        scale = 1 / (key.shape[-1] ** 0.5)

    H_q = query.shape[-3]
    H = value.shape[-3]
    n_repeats = H_q // H
    B = query.shape[:-3]
    L = query.shape[-2]
    S = key.shape[-2]
    D = query.shape[-1]

    query = query.reshape(*B, H, n_repeats,  L, D)
    key = key.reshape(*B, H, 1, S, D)
    value = value.reshape(*B, H, 1, S, D)
    scores = (query @ mx.swapaxes(key, -2, -1)) * scale

    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype)
        mask = mx.broadcast_to(mask, (*B, H_q, L, S))
        mask = mask.reshape(*B, H, n_repeats, L, S)
        scores = scores + mask

    result = softmax(scores, -1) @ value
    return result.reshape(*B, H_q, L, D)

def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    from extensions import tiny_llm_ext

    *B, H_q, L, D = query.shape
    _, H, S, _ = key.shape
    assert H_q % H == 0, "H_q must be divisible by H"

    N = reduce(lambda x, y: x * y, B, 1) * H_q

    query = query.reshape(N, L, D)
    key = key.reshape(-1, S, D)
    value = value.reshape(-1, S, D)

    query = mx.contiguous(query)
    key = mx.contiguous(key)
    value = mx.contiguous(value)

    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, query.dtype)
        mask = mx.broadcast_to(mask, (N, L, S))
        mask = mx.contiguous(mask)
    else:
        mask = mx.reshape(
            mx.broadcast_to(mx.zeros((L, S)), (*B, H_q, L, S)), (N, L, S)
        ).astype(mx.float32)
        mask = mx.contiguous(mask)
    
    scale = mx.rsqrt(D) if scale is None else mx.array(scale).astype(query.dtype)
    
    result = tiny_llm_ext.flash_attention(query, key, value, scale, mask, H_q, H)

    return mx.contiguous(result.reshape(*B, H_q, L, D))




