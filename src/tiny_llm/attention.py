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
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
