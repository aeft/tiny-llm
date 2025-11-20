
#include "quantized_matmul.h"

#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, const int group_size, const int bits,
                           const mx::array &a, const mx::array &b, const bool transpose_b, mx::StreamOrDevice s) {
    assert(transpose_b);

    auto out_shape = {a.shape()[0], b.shape()[0]};
    auto out_dtype = a.dtype();
    return mx::array(out_shape, out_dtype, std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits),
                     {scales, biases, a, b});
}

// CPU implementation
template <typename T>
void quantized_matmul_impl(const mx::array &scales, const mx::array &biases, const mx::array &a, const mx::array &b,
                           mx::array &out, int group_size, int bits, mx::Stream stream) {
    assert(bits == 4);
    assert(group_size == 64);
    assert(b.dtype() == mx::uint32);
    assert(scales.dtype() == out.dtype());
    assert(biases.dtype() == out.dtype());
    assert(a.dtype() == out.dtype());

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    encoder.dispatch([scales_ptr = scales.data<T>(), biases_ptr = biases.data<T>(), a_ptr = a.data<T>(),
                      b_ptr = b.data<uint32_t>(), out_ptr = out.data<T>(), M = a.shape()[0], N = a.shape()[1],
                      K = b.shape()[0], group_size = group_size, bits = bits]() {
        const int group_per_row = N / group_size;
        const int nums_per_item = 32 / bits;  // how many numbers are packed into a uint32_t
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float sum = 0;  // Use float32 for accumulation
                for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
                    auto scale = scales_ptr[k * group_per_row + group_idx];
                    auto bias = biases_ptr[k * group_per_row + group_idx];
                    for (int item_idx = 0; item_idx < group_size; item_idx += 1) {
                        int j = group_idx * group_size + item_idx;
                        // Get the packed value from the b array
                        int pack_idx = j / nums_per_item;
                        int bit_offset = (j % nums_per_item) * bits;
                        uint32_t packed = b_ptr[k * (N / nums_per_item) + pack_idx];
                        uint8_t quantized_value = (packed >> bit_offset) & ((1 << bits) - 1);
                        // Convert the quantized value to a T
                        T b_value = T(quantized_value) * scale + bias;
                        T a_value = a_ptr[i * N + j];
                        // Accumulate the result
                        sum += float(a_value) * float(b_value);
                    }
                }
                out_ptr[i * K + k] = T(sum);
            }
        }
    });
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    // Dispatch to the correct dtype
    if (out.dtype() == mx::float16) {
        quantized_matmul_impl<mx::float16_t>(scales, biases, a, b, out, group_size_, bits_, stream());
    } else if (out.dtype() == mx::bfloat16) {
        quantized_matmul_impl<mx::bfloat16_t>(scales, biases, a, b, out, group_size_, bits_, stream());
    } else {
        throw std::runtime_error("Unsupported dtype");
    }
}

#ifdef _METAL_
void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];
    auto M = a.shape()[0];
    auto N = a.shape()[1];
    auto K = b.shape()[0];
    auto group_size = group_size_;
    auto bits = bits_;
    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto library = d.get_library("tiny_llm_ext");
    const char *kernel_name;
    if (a.dtype() == mx::float16) {
        kernel_name = "quantized_matmul_w4a16_g64_f16";
    } else if (a.dtype() == mx::bfloat16) {
        kernel_name = "quantized_matmul_w4a16_g64_bf16";
    } else {
        throw std::runtime_error("quantized_matmul: a must be float16 or bfloat16");
    }
    auto kernel = d.get_kernel(kernel_name, library);

    // Prepare to encode kernel
    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);
    compute_encoder.set_bytes(group_size, 8);
    compute_encoder.set_bytes(bits, 9);

    // Configure threads
    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const int x_size = 32;
    const int y_size = tgp_size / x_size;
    if (tgp_size < x_size * y_size) {
        throw std::runtime_error("quantized_matmul: tgp_size must be larger than x*y");
    }
    MTL::Size threads_per_group = MTL::Size(x_size, y_size, 1);
    MTL::Size grid_size = MTL::Size(M, K, 1);

    // Dispatch the kernel
    compute_encoder.dispatch_threads(grid_size, threads_per_group);
}
#else
void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("QuantizedMatmul: Metal not available");
}
#endif

}  // namespace tiny_llm_ext