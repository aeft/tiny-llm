import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        
        if top_k is not None and top_k > 0:
            mask_indices = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[:, top_k:]
            batch_indices = mx.arange(logprobs.shape[0])[:, None]
            # Mask out different token sets for each output distribution in the batch
            logprobs[batch_indices, mask_indices] = -mx.inf
        
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = mx.take_along_axis(logprobs, sorted_idx, axis=-1)
            sorted_probs = mx.softmax(sorted_logprobs, axis=-1)
            cumsum = mx.cumsum(sorted_probs, axis=-1)
            # Keep tokens up to and including the first token where cumulative probability exceeds top_p
            mask_elements = cumsum - sorted_probs <= top_p
            # Mask out tokens that are not kept
            masked_sorted_logprobs = mx.where(mask_elements, sorted_logprobs, -mx.inf)
            # Scatter back to original vocabulary positions
            inverse_idx = mx.argsort(sorted_idx, axis=-1)
            logprobs = mx.take_along_axis(masked_sorted_logprobs, inverse_idx, axis=-1)

        return mx.random.categorical(logprobs / temp, axis=-1)

    return sample
