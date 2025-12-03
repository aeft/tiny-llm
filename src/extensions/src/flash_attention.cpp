#include "flash_attention.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

mx::array flash_attention(const mx::array &q, const mx::array &k, const mx::array &v, const float scale,
                          const mx::array &mask, const int num_heads, const int num_kv_heads, mx::StreamOrDevice s) {
    assert(q.dtype() == mx::float32 && k.dtype() == mx::float32 && v.dtype() == mx::float32 &&
           mask.dtype() == mx::float32);
    assert(q.shape().size() == 3 && k.shape().size() == 3 && v.shape().size() == 3 && mask.shape().size() == 3);
    assert(num_heads % num_kv_heads == 0);
    assert(q.shape()[0] % num_heads == 0);
    assert(k.shape()[0] % num_kv_heads == 0);
    assert(v.shape()[0] % num_kv_heads == 0);
    assert(q.shape()[2] == k.shape()[2] && q.shape()[2] == v.shape()[2]);
    assert(mask.shape()[0] == q.shape()[0] && mask.shape()[1] == q.shape()[1] && mask.shape()[2] == k.shape()[1]);
    return mx::array(q.shape(), mx::float32,
                     std::make_shared<FlashAttention>(to_stream(s), scale, num_kv_heads, num_heads), {q, k, v, mask});
}


// Intuition:
// From an algorithmic perspective, each row q_i of the attention matrix is fully independent:
// its output depends only on q_i and all of k/v, and does not depend on any other row.
// In the implementation we group multiple rows into Br-sized row blocks for efficiency,
// but this is purely a batching optimization—the underlying computation is still
// conceptually “row-by-row.”
// Therefore we can focus on the logic for a single row q_i.
//
// For a given row q_i, the inner loop scans all k/v column blocks from left to right.
// When processing the j-th k/v block, we obtain only the corresponding slice of
// the attention score matrix S (the local s_ij values).
// From these local scores we compute the local p_ij (the unnormalized softmax terms),
// and use them to partially accumulate contributions into the output o_i.
//
// At this point o_i is “incomplete” in two ways:
//   (1) The softmax denominator l_i is not yet known, since it requires the sum of
//       all p_ij across the entire row.
//   (2) More importantly, o_i is also missing contributions from the k/v blocks
//       that have not yet been processed.
//
// As we process additional k/v blocks, the remaining columns of this row in the
// full attention matrix are gradually filled in, and their corresponding
// p_ij · v_j contributions are accumulated into o_i. The softmax denominator l_i
// is also incrementally updated.
//
// In the actual implementation, numerical stability must be maintained during this
// block-wise accumulation. Each time we compute new local scores s_ij, we update the
// running maximum m_i of the row. Later blocks may contain larger values, and without
// re-aligning earlier accumulated quantities, the p_ij and o_i could overflow or become
// inconsistent in scale.
// Therefore the algorithm tracks the change m_i_diff, and rescales the previously
// accumulated l_i and o_i using exp(m_i_diff), ensuring that contributions from all
// column blocks remain on the same numerical scale.
//
// After the final k/v block has been processed, all contributions for this row are
// complete and the softmax denominator l_i is fully determined. Normalizing the
// accumulated output as o_i / l_i yields the final attention result.
void FlashAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    // std::cerr << "========== FlashAttention::eval_cpu CALLED ==========" << std::endl;
    // std::cerr.flush();
    auto &q = inputs[0];
    auto &k = inputs[1];
    auto &v = inputs[2];
    auto &mask = inputs[3];
    auto &out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(mask);
    encoder.set_output_array(out);
    encoder.dispatch([q = mx::array::unsafe_weak_copy(q), k = mx::array::unsafe_weak_copy(k),
                      v = mx::array::unsafe_weak_copy(v), mask = mx::array::unsafe_weak_copy(mask), scale = scale_,
                      num_kv_heads = num_kv_heads_, num_heads = num_heads_, out_ptr = out.data<float>()]() {
        const int B = q.shape()[0];
        const int L = q.shape()[1];
        const int D = q.shape()[2];
        const int S = k.shape()[1];

        const float *q_ptr = q.data<float>();
        const float *k_ptr = k.data<float>();
        const float *v_ptr = v.data<float>();
        const float *mask_ptr = mask.data<float>();

        // block size for q
        const int Br = 32;  // TODO: hardcoded for now
        // block size for k/v
        const int Bc = 32;  // TODO: hardcoded for now
        // number of blocks for q
        const int Tr = (L + Br - 1) / Br;
        // number of blocks for k/v
        const int Tc = (S + Bc - 1) / Bc;

        const int n_repeats = num_heads / num_kv_heads;

        // Note in the standard flash attention algorithm, the middle loop is over Tc.
        // loop over batches
        for (int n = 0; n < B; n++) {
            const float *q_n = q_ptr + n * L * D;
            const float *k_n = k_ptr + int(n / n_repeats) * S * D;
            const float *v_n = v_ptr + int(n / n_repeats) * S * D;
            const float *mask_n = mask_ptr + n * L * S;

            // loop over rows of q
            for (int i = 0; i < Tr; i++) {
                std::vector<float> q_i(Br * D, 0.0);
                int br_upper_bound = std::min(L - i * Br, Br);
                // Load Qi
                for (int a = 0; a < br_upper_bound; a++) {
                    for (int b = 0; b < D; b++) {
                        int q_idx = (i * Br + a) * D + b;
                        q_i[a * D + b] = q_n[q_idx];
                    }
                }
                std::vector<float> o_i(Br * D, 0.0);
                std::vector<float> l_i(Br, 0.0);
                std::vector<float> m_i(Br, -std::numeric_limits<float>::infinity());

                // loop over columns of k/v
                for (int j = 0; j < Tc; j++) {
                    int bc_upper_bound = std::min(S - j * Bc, Bc);
                    std::vector<float> k_j(Bc * D, 0.0);
                    std::vector<float> v_j(Bc * D, 0.0);
                    // Load k_j and v_j
                    for (int a = 0; a < bc_upper_bound; a++) {
                        for (int b = 0; b < D; b++) {
                            int kv_idx = (j * Bc + a) * D + b;
                            k_j[a * D + b] = k_n[kv_idx];
                            v_j[a * D + b] = v_n[kv_idx];
                        }
                    }

                    // Compute s_ij = q_i * k_j^T
                    std::vector<float> s_i(Br * Bc, 0.0);
                    for (int a = 0; a < br_upper_bound; a++) {
                        for (int b = 0; b < bc_upper_bound; b++) {
                            for (int c = 0; c < D; c++) {
                                s_i[a * Bc + b] += q_i[a * D + c] * k_j[b * D + c];
                            }
                            // Apply scale and mask
                            s_i[a * Bc + b] *= scale;
                            s_i[a * Bc + b] += mask_n[(i * Br + a) * S + (j * Bc + b)];
                        }
                    }

                    // Compute m_i = max(m_i, rowmax(s_i))
                    std::vector<float> m_i_diff(Br, 0.0);
                    for (int a = 0; a < br_upper_bound; a++) {
                        float rowmax = -std::numeric_limits<float>::infinity();
                        for (int b = 0; b < bc_upper_bound; b++) {
                            rowmax = std::max(rowmax, s_i[a * Bc + b]);
                        }
                        float max = std::max(m_i[a], rowmax);
                        m_i_diff[a] = m_i[a] - max;
                        m_i[a] = max;
                    }

                    // Compute p_ij = exp(s_ij - m_i)
                    std::vector<float> p(Br * Bc, 0.0);
                    for (int a = 0; a < br_upper_bound; a++) {
                        for (int b = 0; b < bc_upper_bound; b++) {
                            p[a * Bc + b] = std::exp(s_i[a * Bc + b] - m_i[a]);
                        }
                    }

                    // Compute l_i = l_i + sum(p_ij)
                    for (int a = 0; a < br_upper_bound; a++) {
                        float rowsum = 0.0;
                        for (int b = 0; b < bc_upper_bound; b++) {
                            rowsum += p[a * Bc + b];
                        }
                        // cancel out the m_i[a] from the previous iteration
                        l_i[a] = l_i[a] * std::exp(m_i_diff[a]) + rowsum;
                    }

                    // Compute o_ik = exp(m_i_diff) * o_ik + p_ij @ v_jk
                    for (int a = 0; a < br_upper_bound; a++) {
                        for (int c = 0; c < D; c++) {
                            float pv = 0.0;
                            for (int b = 0; b < bc_upper_bound; b++) {
                                pv += p[a * Bc + b] * v_j[b * D + c];
                            }
                            // cancel out the m_i[a] from the previous iteration
                            o_i[a * D + c] = o_i[a * D + c] * std::exp(m_i_diff[a]) + pv;
                        }
                    }
                }
                // o_i = o_i / l_i and store it to out
                for (int a = 0; a < br_upper_bound; a++) {
                    for (int b = 0; b < D; b++) {
                        o_i[a * D + b] /= l_i[a];
                        int out_idx = n * L * D + (i * Br + a) * D + b;
                        out_ptr[out_idx] = o_i[a * D + b];
                    }
                }
            }
        }
    });
}

void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    // std::cerr << "========== FlashAttention::eval_gpu CALLED ==========" << std::endl;
    // std::cerr.flush();
#ifdef _METAL_
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];
    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel("flash_attention_f32_e128", library);
    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(mask, 3);
    compute_encoder.set_output_array(out, 4);

    const int N = q.shape()[0];
    const int L = q.shape()[1];
    const int S = k.shape()[1];
    const int D = q.shape()[2];
    const int Br = 32;
    const int Bc = 32;
    const int Tr = (L + Br - 1) / Br;
    const int Tc = (S + Bc - 1) / Bc;

    compute_encoder.set_bytes(N, 5);
    compute_encoder.set_bytes(L, 6);
    compute_encoder.set_bytes(S, 7);
    compute_encoder.set_bytes(D, 8);
    compute_encoder.set_bytes(num_kv_heads_, 9);
    compute_encoder.set_bytes(num_heads_, 10);
    compute_encoder.set_bytes(scale_, 11);
    compute_encoder.set_bytes(Br, 12);
    compute_encoder.set_bytes(Bc, 13);
    compute_encoder.set_bytes(Tr, 14);
    compute_encoder.set_bytes(Tc, 15);

    size_t simd_width = kernel->threadExecutionWidth(); // threads per simdgroup

    MTL::Size num_threadgroups = MTL::Size(N, Tr, 1);
    MTL::Size num_threads_per_group = MTL::Size(Br, simd_width, 1);
    compute_encoder.dispatch_threadgroups(num_threadgroups, num_threads_per_group);

#else
    throw std::runtime_error("FlashAttention GPU implementation not yet available");
#endif
}

}  // namespace tiny_llm_ext