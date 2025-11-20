#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g64(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    device const int &group_size [[buffer(8)]],
    device const int &bits [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int i = gid.x;
    const int k = gid.y;

    if (i >= M || k >= K) {
        return;
    }

    float sum = 0;  // Use float32 for accumulation
    int group_per_row = N / group_size;
    int nums_per_item = 32 / bits;  // how many numbers are packed into a uint32_t
    for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
        auto scale = scales[k * group_per_row + group_idx];
        auto bias = biases[k * group_per_row + group_idx];
        for (int item_idx = 0; item_idx < group_size; item_idx += 1) {
            int j = group_idx * group_size + item_idx;
            int pack_idx = j / nums_per_item;
            int bit_offset = (j % nums_per_item) * bits;
            uint32_t packed = b[k * (N / nums_per_item) + pack_idx];
            uint8_t quantized_value = (packed >> bit_offset) & ((1 << bits) - 1);
            // Convert the quantized value to a float
            T b_value = T(quantized_value) * scale + bias;
            T a_value = a[i * N + j];
            // Accumulate the result
            sum += float(a_value) * float(b_value);
        }
    }
    out[i * K + k] = T(sum);
}

// Instantiate for float16 and bfloat16
instantiate_kernel("quantized_matmul_w4a16_g64_f16", quantized_matmul_w4a16_g64, float16_t);
instantiate_kernel("quantized_matmul_w4a16_g64_bf16", quantized_matmul_w4a16_g64, bfloat16_t);