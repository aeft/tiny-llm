#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

[[kernel]] void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int &N [[buffer(5)]], // N = B * H_q
    device const int &L [[buffer(6)]], // L = seq_len
    device const int &S [[buffer(7)]], // S = seq_len_kv
    device const int &D [[buffer(8)]], // D = embedding dim
    device const int &num_kv_heads [[buffer(9)]],
    device const int &num_heads [[buffer(10)]],
    device const float &scale [[buffer(11)]],
    device const int &Br [[buffer(12)]], // block size for q
    device const int &Bc [[buffer(13)]], // block size for k/v
    device const int &Tr [[buffer(14)]], // number of blocks for q
    device const int &Tc [[buffer(15)]], // number of blocks for k/v
    uint2 group_id [[threadgroup_position_in_grid]], // (n, i) means the n-th head and the i-th block of q
    uint simd_gid [[simdgroup_index_in_threadgroup]], // a (row in block)
    uint simd_tid [[thread_index_in_simdgroup]]) { // b (column in block)

    int n = group_id.x;
    int i = group_id.y; // loop over Tr
    int a = simd_gid; // max=Br
    int b = simd_tid; // max=Bc

    bool is_i_in_range = i * Br + a < L && a < Br;
    // We cannot return directly since we need to sync the threadgroup memory.
    // Since we don't return directly, we should always check the boundry when accessing memory.
    // if (!is_i_in_range) return;

    const int n_repeats = num_heads / num_kv_heads;
    device const float *q_ptr = q + n * L * D + i * Br * D;
    device float *out_ptr = out + n * L * D + i * Br * D;
    // Loop over k/v blocks
    device const float *k_ptr_base = k + (n / n_repeats) * S * D;
    device const float *v_ptr_base = v + (n / n_repeats) * S * D;

    // Allocate shared memory and initialize
    threadgroup float o_i[32 * 128]; // MAX_D=128, Br=32
    threadgroup float q_local[32][128];
    if (simd_tid == 0) {
        for (int c = 0; c < D; c++) {
            o_i[a * D + c] = 0;
            q_local[a][c] = q_ptr[a * D + c];
        }
    }

    float m_i = -1e9;
    float l_i = 0;

    for (int j = 0; j < Tc; j++) {
        bool is_j_in_range = b < Bc && j * Bc + b < S;
        device const float *k_ptr = k_ptr_base + j * Bc * D;
        device const float *v_ptr = v_ptr_base + j * Bc * D;

        // Compute s_ij = q_i * k_j^T
        float s = 0.0;
        for (int c = 0; c < D; c++) {
            if (is_i_in_range && is_j_in_range) {
                s += q_local[a][c] * k_ptr[b * D + c];
            }
        }
        s *= scale;
        // Apply mask
        if (is_i_in_range && is_j_in_range) {
            s += mask[n * L * S + (i * Br + a) * S + (j * Bc + b)];
        } else {
            s = -1e9;
        }

        // Update m_i
        // Actually, all threads in a simdgroup share the same m_i, because they have gone through the same process.
        float rowmax = simd_max(s);
        float new_max = max(m_i, rowmax);
        float m_i_diff = m_i - new_max;
        m_i = new_max;

        // Compute p_ij = exp(s_ij - m_i)
        float p = 0.0;
        if (is_i_in_range && is_j_in_range) {
            p = exp(s - m_i);
        }

        // Compute l_i = l_i + sum(p_ij)
        float rowsum = simd_sum(p);
        l_i = l_i * exp(m_i_diff) + rowsum; // same here: all threads in a simdgroup share the same l_i.

        // sync here becasue we need to write to o_i  
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute o_ik = exp(m_i_diff) * o_ik + p_ij @ v_jk
        for (int c = 0; c < D; c++) {
            float pv = 0.0;
            if (is_i_in_range && is_j_in_range) {
                pv = p * v_ptr[b * D + c];
            }
            float res = simd_sum(pv);
            if (simd_tid == 0 && is_i_in_range) {
                o_i[a * D + c] = exp(m_i_diff) * o_i[a * D + c] + res;
            }
        }
    }
    // We first write to threadgroup memory (o_i), and then sync to global memory (out).
    if (simd_tid == 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < D; c++) {
            if (is_i_in_range && n < N) {
                out_ptr[a * D + c] = o_i[a * D + c] / l_i;
            }
        }
    }
}