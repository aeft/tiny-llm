import pytest
import mlx.core as mx
from tiny_llm.sampler import make_sampler


class TestSampler:
    """Test different sampling strategies"""
    
    def test_greedy_sampling(self):
        """Test greedy sampling (temp=0) always picks highest probability"""
        sampler = make_sampler(temp=0, top_p=None, top_k=None)
        
        # Batch of 2, vocab size 5
        logprobs = mx.array([
            [1.0, 3.0, 2.0, 0.5, 0.1],  # max at index 1
            [0.1, 0.2, 5.0, 1.0, 0.5],  # max at index 2
        ])
        
        result = sampler(logprobs)
        assert result.shape == (2,)
        assert result[0].item() == 1
        assert result[1].item() == 2
    
    def test_temperature_sampling(self):
        """Test temperature sampling produces valid token indices"""
        sampler = make_sampler(temp=1.0, top_p=None, top_k=None)
        
        logprobs = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        # Run multiple times to check randomness
        results = [sampler(logprobs).item() for _ in range(10)]
        
        # Should produce valid indices
        assert all(0 <= r < 5 for r in results)
        # With temp=1.0 and clear preferences, should show variety
        assert len(set(results)) >= 2
    
    def test_top_k_sampling(self):
        """Test top-k sampling only samples from top-k tokens"""
        mx.random.seed(42)
        sampler = make_sampler(temp=1.0, top_p=None, top_k=2)
        
        # Clear preference for indices 3 and 4
        logprobs = mx.array([[0.0, 0.1, 0.2, 10.0, 9.0]])
        
        # Sample many times
        results = [sampler(logprobs).item() for _ in range(50)]
        
        # Should only sample from top-2 (indices 3 and 4)
        unique_results = set(results)
        assert unique_results == {3, 4}
    
    def test_top_p_sampling(self):
        """Test top-p (nucleus) sampling respects probability threshold"""
        mx.random.seed(42)
        sampler = make_sampler(temp=1.0, top_p=0.9, top_k=None)
        
        # Create distribution where one token dominates
        logprobs = mx.array([[10.0, 5.0, 1.0, 0.0, 0.0]])
        
        # Sample many times
        results = [sampler(logprobs).item() for _ in range(100)]
        
        # Should produce valid indices
        assert all(0 <= r < 5 for r in results)
        # Most samples should be from high probability tokens
        assert results.count(0) > 50  # Dominant token
    
    def test_batch_sampling(self):
        """Test sampling works correctly with batches"""
        sampler = make_sampler(temp=0, top_p=None, top_k=None)
        
        # Batch of 3
        logprobs = mx.array([
            [5.0, 1.0, 2.0],
            [1.0, 5.0, 2.0],
            [1.0, 2.0, 5.0],
        ])
        
        result = sampler(logprobs)
        assert result.shape == (3,)
        assert result[0].item() == 0
        assert result[1].item() == 1
        assert result[2].item() == 2
    
    def test_combined_top_k_and_top_p(self):
        """Test combining top-k and top-p"""
        mx.random.seed(42)
        sampler = make_sampler(temp=1.0, top_p=0.6, top_k=4)
        
        # Create a distribution where:
        # - top-k=4 would allow indices 0,1,2,3
        # - top-p=0.6 should further restrict to fewer tokens
        logprobs = mx.array([[5.0, 4.0, 3.5, 2.0, 0.0, 0.0]])
        
        results = [sampler(logprobs).item() for _ in range(100)]
        unique_results = set(results)
        
        # With cumsum ~[0.604, 0.826, ...], only index 0 should remain after top-p
        # (index 0 is always kept even if cumsum[0] > top_p)
        assert unique_results == {0}, f"top-p should have kept only index 0, got: {unique_results}"
    
    def test_top_p_scatter_back_issue(self):
        """top-p should scatter values back to original vocabulary positions"""
        mx.random.seed(123)
        sampler = make_sampler(temp=1.0, top_p=0.7, top_k=None)
        
        # Create a distribution where middle indices have highest probabilities
        # After softmax of [0, 0, 5.0, 4.5, 4.0, 0]:
        #   probs ≈ [0.003, 0.003, 0.501, 0.304, 0.184, 0.003]
        # Sorted (descending): [0.501, 0.304, 0.184, ...]
        # Cumsum: [0.501, 0.805, 0.989, ...]
        # With top_p=0.7, first 2 tokens are kept (cumsum[1] > 0.7)
        # These correspond to original indices [2, 3]
        logprobs = mx.array([[0.0, 0.0, 5.0, 4.5, 4.0, 0.0]])
        
        # Sample many times
        results = [sampler(logprobs).item() for _ in range(100)]
        unique_results = set(results)
        
        # With broken implementation, the sorted logprobs aren't scattered back,
        # so we might sample from wrong positions (e.g., indices 0, 1 instead of 2, 3)
        assert unique_results == {2, 3}, \
            f"Expected to sample from indices 2,3 but got: {unique_results}. " \
            f"Issue: logprobs not scattered back to original positions after top-p masking."
    
    def test_top_p_boundary_condition_issue(self):
        """When first token has prob >= top_p, it should be kept"""
        mx.random.seed(1)
        sampler = make_sampler(temp=1.0, top_p=0.5, top_k=None)
        
        # Create a distribution where the highest probability token (at index 1) has prob > top_p
        # After softmax: [~0.0003, ~0.999, ~0.0003, ~0.0001, ~0.0001]
        logprobs = mx.array([[2.0, 10.0, 2.0, 1.0, 1.0]])
        
        # After sorting by descending probability: [10.0, 2.0, 2.0, 1.0, 1.0]
        # This corresponds to original indices: [1, 0, 2, 3, 4]
        # With top_p=0.5, cumsum[0] will be ~0.999 which is > 0.5
        
        results = [sampler(logprobs).item() for _ in range(50)]
        unique_results = set(results)
        
        # Should sample mostly from index 1 (the dominant token at original position)
        assert 1 in unique_results, \
            f"Expected to sample index 1, but got: {unique_results}. " \
            f"Issue: First token not kept when its probability exceeds top_p."
        
        # Most samples should be index 1
        assert results.count(1) > 40, \
            f"Expected mostly index 1, but got distribution: {unique_results}. " \
            f"Count of 1: {results.count(1)}/50"
    
    def test_top_k_batch_indexing_issue(self):
        """top-k batch indexing"""
        mx.random.seed(42)
        sampler = make_sampler(temp=1.0, top_p=None, top_k=2)
        
        # Batch of 2 sequences with different top-k tokens
        # Batch 0: highest are indices 0 (10.0) and 1 (9.0)
        # Batch 1: highest are indices 3 (10.0) and 4 (9.0)
        logprobs = mx.array([
            [10.0, 9.0, 0.0, 0.0, 0.0],  # Should only sample from indices 0, 1
            [0.0, 0.0, 0.0, 10.0, 9.0],  # Should only sample from indices 3, 4
        ])
        
        # Sample many times for each batch
        results_batch_0 = []
        results_batch_1 = []
        for _ in range(50):
            result = sampler(logprobs)
            results_batch_0.append(result[0].item())
            results_batch_1.append(result[1].item())
        
        unique_batch_0 = set(results_batch_0)
        unique_batch_1 = set(results_batch_1)
        
        # Each batch should only sample from its own top-k tokens
        assert unique_batch_0 == {0, 1}, \
            f"Batch 0 should only sample from indices 0,1 but got: {unique_batch_0}. " \
            f"Issue: top-k batch indexing is broken - mask_elements indexing doesn't work correctly for batches."
        
        assert unique_batch_1 == {3, 4}, \
            f"Batch 1 should only sample from indices 3,4 but got: {unique_batch_1}. " \
            f"Issue: top-k batch indexing is broken - mask_elements indexing doesn't work correctly for batches."

