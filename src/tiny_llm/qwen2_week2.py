import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_flash_attention = use_flash_attention

        self.rope = RoPE(self.hidden_size // self.num_heads, max_seq_len, theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:

        B, L_new, E = x.shape
        D = E // self.num_heads

        projection_q = linear(x, dequantize_linear(self.wq), bias=self.bq).reshape(
            B, L_new, self.num_heads, D
        )
        projection_k = linear(x, dequantize_linear(self.wk), bias=self.bk).reshape(
            B, L_new, self.num_kv_heads, D
        )
        projection_v = linear(x, dequantize_linear(self.wv), bias=self.bv).reshape(
            B, L_new, self.num_kv_heads, D)

        if isinstance(offsets, int):
            offset_slice = [slice(int(offsets), int(offsets + L_new))]
        else:
            offset_slice = [slice(int(i), int(i + L_new)) for i in offsets]

        # Note q and k are incrementally applied, so we need to apply the offset to both q and k
        projection_q = self.rope(projection_q, offset=offset_slice)
        projection_k = self.rope(projection_k, offset=offset_slice)

        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)

        projection_k, projection_v, _, mask = cache.update_and_fetch(projection_k, projection_v, mask_length=L_new, mask=mask)

        x = scaled_dot_product_attention_grouped(
            projection_q.astype(mx.float32),
            projection_k.astype(mx.float32),
            projection_v.astype(mx.float32),
            mask=mask,
        ).astype(x.dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L_new, self.hidden_size)
        return linear(x, dequantize_linear(self.wo))



class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate = linear(x, dequantize_linear(self.w_gate))
        up = linear(x, dequantize_linear(self.w_up))
        return linear(silu(gate) * up, dequantize_linear(self.w_down))


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_flash_attention = use_flash_attention

        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, eps=rms_norm_eps)
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
            use_flash_attention=use_flash_attention,
        )
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        offset: int | list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x_norm = self.input_layernorm(x)
        x_attn = self.self_attn(x_norm, offset, cache, mask)
        x = x + x_attn
        x_norm = self.post_attention_layernorm(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp
        return x


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.mlx_model = mlx_model

        self.embedding = Embedding(
            vocab_size=mlx_model.args.vocab_size,
            embedding_dim=mlx_model.args.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens),
        )

        self.transformer_blocks = []
        for i in range(mlx_model.args.num_hidden_layers):
            wq = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].self_attn.q_proj)
            wk = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].self_attn.k_proj)
            wv = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].self_attn.v_proj)
            wo = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].mlp.up_proj)
            w_down = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].mlp.down_proj)
            w_input_layernorm = mlx_model.model.layers[i].input_layernorm.weight
            w_post_attention_layernorm = mlx_model.model.layers[i].post_attention_layernorm.weight
            transformer_block = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias,
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias,
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=w_input_layernorm,
                w_post_attention_layernorm=w_post_attention_layernorm,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
                use_flash_attention=enable_flash_attn,
            )
            self.transformer_blocks.append(transformer_block)

        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )

        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = QuantizedWeights.from_mlx_layer(mlx_model.lm_head)
        else:
            self.w_lm_head = None

    def __call__(
        self,
        inputs: mx.array,
        offset: int | list[int],
        cache: list[TinyKvCache],
    ) -> mx.array:
        h = self.embedding(inputs)
        for i, layer in enumerate(self.transformer_blocks):
            h = layer(h, offset, cache[i], mask="causal")
        h = self.norm(h)
        if self.w_lm_head is not None:
            return linear(h, dequantize_linear(self.w_lm_head))
        else:
            return self.embedding.as_linear(h)
