
#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

mx::array quantized_matmul(
    const mx::array &scales,
    const mx::array &biases,
    const int group_size,
    const int bits,
    const mx::array &a,
    const mx::array &b,
    const bool transpose_b,
    mx::StreamOrDevice s = {}
);

class QuantizedMatmul : public mx::Primitive {
public:
    explicit QuantizedMatmul(mx::Stream stream, const int group_size, const int bits)
        : mx::Primitive(stream), group_size_(group_size), bits_(bits) {};

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    const char *name() const override { return "QuantizedMatmul"; }
    
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array> &inputs,
        const std::vector<int> &axes) override {
        throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
    }

private:
    int group_size_;
    int bits_;
};

}  // namespace tiny_llm_ext