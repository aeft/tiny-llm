import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        half_dims = dims // 2
        self.half_dims = half_dims

        # inner: [0/half, 1/half, ..., (half-1)/half]
        inner = mx.arange(0, half_dims, dtype=mx.float32) / half_dims  # (half,)

        # inv_freq: base^{-inner}
        inv_freq = mx.power(base, -inner)  # (half,)

        # t: [0, 1, 2, ..., seq_len-1]
        t = mx.arange(seq_len)  # (seq_len,)

        # angle matrix ang[pos, pair] = t[pos] * inv_freq[pair]
        ang = mx.outer(t, inv_freq)  # (seq_len, half)

        # cosine and sine frequency tables
        self.cos_freqs = mx.cos(ang)  # (seq_len, half)
        self.sin_freqs = mx.sin(ang)  # (seq_len, half)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape
        if offset is None:
            cos_basis = self.cos_freqs[:S, :]    # (S, half)
            sin_basis = self.sin_freqs[:S, :]    # (S, half)
        else:
            assert isinstance(offset, slice)
            start, stop = offset.start, offset.stop
            assert stop - start == S, f"offset must be of length {S}"
            cos_basis = self.cos_freqs[start:stop, :]
            sin_basis = self.sin_freqs[start:stop, :]
        
        cos_basis = cos_basis.reshape(1, S, 1, self.half_dims)
        sin_basis = sin_basis.reshape(1, S, 1, self.half_dims)
        
        if self.traditional:
            x = x.reshape(N, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
            real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
            imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        else:
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]
            real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
            imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        return y.astype(x.dtype)
        

