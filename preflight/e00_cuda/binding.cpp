#include <cstdint>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "xor_kernel.h"

namespace {

void xor_out(const at::Tensor& input, at::Tensor& output) {
  TORCH_CHECK(input.layout() == c10::kStrided, "input must be strided");
  TORCH_CHECK(output.layout() == c10::kStrided, "output must be strided");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(
      input.scalar_type() == at::kInt,
      "input must have dtype torch.int32");
  TORCH_CHECK(
      output.scalar_type() == at::kInt,
      "output must have dtype torch.int32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(
      input.sizes() == output.sizes(),
      "input and output must have identical shapes");
  TORCH_CHECK(
      input.device() == output.device(),
      "input and output must be on the same CUDA device");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  e00_launch_xor_out(
      input.data_ptr<std::int32_t>(),
      output.data_ptr<std::int32_t>(),
      input.numel(),
      stream);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "xor_out",
      &xor_out,
      "E00 certification-only out-parameter int32 XOR (CUDA)");
}
