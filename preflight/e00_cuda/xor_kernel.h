#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

void e00_launch_xor_out(
    const std::int32_t* input,
    std::int32_t* output,
    std::int64_t numel,
    cudaStream_t stream);
