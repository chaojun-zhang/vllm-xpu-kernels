/**
 * XPU (SYCL/DPC++) implementation of AWQ weight dequantization.
 *
 * Mirrors the CUDA kernel in vllm/csrc/quantization/awq/gemm_kernels.cu.
 *
 * Tensor layouts (AWQ packing):
 *   qweight : [in_c, qout_c]    int32  — 8 int4 nibbles packed per element
 *   scales  : [in_c/G, out_c]   fp16   — per-group scale factors
 *   zeros   : [in_c/G, qout_c]  int32  — 8 int4 nibbles packed per element
 *   output  : [in_c, out_c]     fp16   — dequantized weights
 *
 * Each work-item handles one (row, col) entry and writes 8 fp16 values:
 *   output[row, col*8+i] = (w_nibble_i - z_nibble_i) * scale[row/G, col*8+i]
 */

#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include "ops.h"

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy) {
  TORCH_CHECK(_kernel.dtype() == torch::kInt32,
              "awq_dequantize: qweight must be int32");
  TORCH_CHECK(_scaling_factors.dtype() == torch::kFloat16,
              "awq_dequantize: scales must be fp16");
  TORCH_CHECK(_zeros.dtype() == torch::kInt32,
              "awq_dequantize: zeros must be int32");

  const int in_c = _kernel.size(0);
  const int qout_c = _kernel.size(1);
  const int out_c = qout_c * 8;
  const int G = in_c / _scaling_factors.size(0);

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  torch::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

  const int* qweight = _kernel.data_ptr<int>();
  const sycl::half* scales = reinterpret_cast<const sycl::half*>(
      _scaling_factors.data_ptr<at::Half>());
  const int* zeros = _zeros.data_ptr<int>();
  sycl::half* output =
      reinterpret_cast<sycl::half*>(_de_kernel.data_ptr<at::Half>());

  const int total_wi = in_c * qout_c;

  auto& q = at::xpu::getCurrentXPUStream().queue();
  constexpr int WG_SIZE = 64;
  const int num_wgs = (total_wi + WG_SIZE - 1) / WG_SIZE;

  q.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_wgs * WG_SIZE),
                          sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          const int idx = static_cast<int>(item.get_global_id(0));
          if (idx >= total_wi) return;

          const int col = idx % qout_c;
          const int row = idx / qout_c;
          const int group = row / G;

          const uint32_t w_packed =
              static_cast<uint32_t>(qweight[row * qout_c + col]);
          const uint32_t z_packed =
              static_cast<uint32_t>(zeros[group * qout_c + col]);

          const sycl::half* scale_base = scales + group * out_c + col * 8;
          sycl::half* out_base = output + row * out_c + col * 8;

          // AWQ uses interleaved nibble packing order: [0, 4, 1, 5, 2, 6, 3, 7]
          // reverse_awq_order[i] is the nibble index for output column offset i.
          // This matches CUDA dequantize_s4_to_fp16x2 and Triton AWQ kernels.
          static constexpr int reverse_awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int shift = reverse_awq_order[i] * 4;
            const float w_f =
                static_cast<float>((w_packed >> shift) & 0xFu);
            const float z_f =
                static_cast<float>((z_packed >> shift) & 0xFu);
            const float s_f = static_cast<float>(scale_base[i]);
            out_base[i] = static_cast<sycl::half>((w_f - z_f) * s_f);
          }
        });
  });

  return _de_kernel;
}
