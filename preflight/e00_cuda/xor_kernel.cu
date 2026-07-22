#include "xor_kernel.h"

#include <algorithm>
#include <cstdint>

#include <c10/cuda/CUDAException.h>

namespace {

constexpr std::int32_t kXorMask = 0x5A5A5A5A;
constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kMaximumBlocks = 65535;

__global__ void e00_xor_out_kernel(
    const std::int32_t* input,
    std::int32_t* output,
    std::int64_t numel) {
  const std::int64_t first =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride =
      static_cast<std::int64_t>(blockDim.x) * gridDim.x;

  for (std::int64_t index = first; index < numel; index += stride) {
    output[index] = input[index] ^ kXorMask;
  }
}

}  // namespace

void e00_launch_xor_out(
    const std::int32_t* input,
    std::int32_t* output,
    std::int64_t numel,
    cudaStream_t stream) {
  if (numel == 0) {
    return;
  }

  const std::int64_t required_blocks =
      (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int blocks = static_cast<int>(
      std::min(required_blocks, kMaximumBlocks));
  e00_xor_out_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, output, numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
