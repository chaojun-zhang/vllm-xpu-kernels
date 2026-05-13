/*
 * Adapted from https://github.com/mit-han-lab/llm-awq
 * XPU (SYCL/DPC++) port of AWQ weight dequantization.
 * Mirrors vllm/csrc/quantization/awq/gemm_kernels.cu.
 *
 * Tensor layouts (AWQ packing):
 *   qweight : [in_c, qout_c]    int32  — 8 int4 nibbles packed per element
 *   scales  : [in_c/G, out_c]   fp16   — per-group scale factors
 *   zeros   : [in_c/G, qout_c]  int32  — 8 int4 nibbles packed per element
 *   output  : [in_c, out_c]     fp16   — dequantized weights
 */

#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include "ops.h"

namespace vllm {
namespace awq {

// SYCL port of dequantize_s4_to_fp16x2 from dequantize.cuh.
// Unpacks a packed int32 (8 x int4) into 8 fp16 values stored in out[0..7].
// AWQ interleaved nibble order maps output position i to nibble shifts[i]:
//   out[0]=nibble0, out[1]=nibble4, out[2]=nibble1, out[3]=nibble5,
//   out[4]=nibble2, out[5]=nibble6, out[6]=nibble3, out[7]=nibble7
inline void dequantize_s4_to_fp16x2(uint32_t source, sycl::half* out) {
  static constexpr uint32_t shifts[8] = {0, 16, 4, 20, 8, 24, 12, 28};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = static_cast<sycl::half>((source >> shifts[i]) & 0xFu);
  }
}

// Kernel functor — mirrors the dequantize_weights __global__ from
// gemm_kernels.cu. 2-D indexing: dim-0 = row (in_c), dim-1 = col (qout_c).
// N_ = total columns = blockDim.x * gridDim.x = qout_c.
class dequantize_weights_kernel {
 private:
  const int* __restrict__ B_;
  const sycl::half* __restrict__ scaling_factors_;
  const int* __restrict__ zeros_;
  sycl::half* __restrict__ C_;
  int G_;
  int N_;

 public:
  dequantize_weights_kernel(const int* B, const sycl::half* scaling_factors,
                            const int* zeros, sycl::half* C, int G, int N)
      : B_(B),
        scaling_factors_(scaling_factors),
        zeros_(zeros),
        C_(C),
        G_(G),
        N_(N) {}

  void operator()(sycl::nd_item<2> item) const {
    const int col = static_cast<int>(item.get_global_id(1));
    const int row = static_cast<int>(item.get_global_id(0));

    sycl::half* C_ptr2 = C_ + 8 * col + 8 * row * N_;

    const uint32_t B_loaded = static_cast<uint32_t>(B_[col + row * N_]);
    const uint32_t zeros_loaded =
        static_cast<uint32_t>(zeros_[col + (row / G_) * N_]);
    const sycl::half* scaling_factors_ptr2 =
        scaling_factors_ + 8 * col + (row / G_) * N_ * 8;

    sycl::half B_loaded_fp16[8];
    sycl::half B_loaded_zero[8];
    dequantize_s4_to_fp16x2(B_loaded, B_loaded_fp16);
    dequantize_s4_to_fp16x2(zeros_loaded, B_loaded_zero);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      C_ptr2[i] = static_cast<sycl::half>(
          (static_cast<float>(B_loaded_fp16[i]) -
           static_cast<float>(B_loaded_zero[i])) *
          static_cast<float>(scaling_factors_ptr2[i]));
    }
  }
};

}  // namespace awq
}  // namespace vllm

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy) {
  const int in_c = _kernel.size(0);
  const int qout_c = _kernel.size(1);
  const int out_c = qout_c * 8;
  const int G = in_c / _scaling_factors.size(0);

  // Mirror CUDA grid-dim computation from gemm_kernels.cu.
  int x_thread = static_cast<int>(thx);
  int y_thread = static_cast<int>(thy);
  int x_blocks = 1;
  int y_blocks = 1;
  if (thx == 0) {
    x_thread = qout_c;
  }
  if (thy == 0) {
    y_thread = in_c;
  }
  if (thx == 0 && thy == 0) {
    x_thread = 8;
    y_thread = 8;
    x_blocks = qout_c / 8;
    y_blocks = in_c / 8;
  }

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  torch::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

  const int* kernel = _kernel.data_ptr<int>();
  sycl::half* de_kernel =
      reinterpret_cast<sycl::half*>(_de_kernel.data_ptr<at::Half>());
  const sycl::half* scaling_factors = reinterpret_cast<const sycl::half*>(
      _scaling_factors.data_ptr<at::Half>());
  const int* zeros = _zeros.data_ptr<int>();

  vllm::awq::dequantize_weights_kernel kfn(kernel, scaling_factors, zeros,
                                           de_kernel, G, qout_c);

  auto& q = at::xpu::getCurrentXPUStream().queue();
  q.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(y_blocks * y_thread, x_blocks * x_thread),
            sycl::range<2>(y_thread, x_thread)),
        kfn);
  });

  return _de_kernel;
}
