// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

/*
 * SYCL kernels for fused quantized RMS normalization on XPU.
 *
 * Two entry points:
 *   - rms_norm_static_fp8_quant
 *   - fused_add_rms_norm_static_fp8_quant
 *
 * Precision design
 * ----------------
 * The kernels are structured to match the CUDA upstream in
 * layernorm_quant_kernels.cu as closely as possible.
 *
 * rms_norm_static_fp8_quant_kernel<VEC_SIZE>  (single unified kernel):
 *   Pass 1 — variance reduction via vllm::vectorize_read_with_alignment,
 *             which handles the unaligned prefix, aligned vec body, and tail
 *             exactly as the CUDA version does.  VEC_SIZE=1 degrades to pure
 *             scalar (same as the CUDA scalar path).
 *   Pass 2 — vectorized normalize + weight + fp8, iterating hidden/VEC_SIZE
 *             vec-elements per thread (mirrors CUDA pass 2).
 *
 * fused_add_rms_norm_static_fp8_quant_kernel<VEC_SIZE>  (single unified kernel):
 *   Pass 1 — fused residual-add + variance via
 *             vllm::vectorize_fused_add_with_alignment, handling unaligned
 *             prefix, aligned vec body, and tail.  VEC_SIZE=1 degrades to
 *             pure scalar.
 *   Pass 2 — same vec loop as rms_norm Pass 2, reading from residual_row
 *             (updated in Pass 1).
 *
 * Both kernels dispatch via calculated_vec_size = gcd(16/sizeof, hidden_size)
 * + VLLM_DISPATCH_VEC_SIZE, mirroring the CUDA upstream dispatch exactly.
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>

#include "dispatch_utils.h"
#include "ops.h"
#include "quantization/fp8/quant_utils.h"
#include "quantization/vectorization.h"
#include "quantization/vectorization_utils.h"
#include "utils.h"

namespace vllm {

// ===================================================================
// Kernel 1: rms_norm_static_fp8_quant
//
// VEC_P1: Pass 1 VEC_SIZE — fixed to 4, matching rms_norm_kernel in
//         layernorm.cpp (also VEC_SIZE=4).  Using a different width
//         changes per-thread partial sums → different reduce_over_group
//         result → different variance → fp8 boundary crossing.
// VEC_P2: Pass 2 VEC_SIZE — gcd(16/sizeof, hidden_size), maximises
//         load/store width.  Pass 2 has no reduction so its width does
//         not affect numerical results.
// ===================================================================
// TODO: Further optimize this kernel.
template <typename scalar_t, int VEC_P1, int VEC_P2, typename fp8_type>
class rms_norm_static_fp8_quant_kernel {
 public:
  rms_norm_static_fp8_quant_kernel(
      fp8_type* __restrict__ out,
      const scalar_t* __restrict__ input,
      const int input_stride,
      const scalar_t* __restrict__ weight,
      const float scale_inv,
      const float epsilon,
      const int num_tokens,
      const int hidden_size,
      sycl::local_accessor<float, 1> s_variance)
      : out_(out), input_(input), input_stride_(input_stride),
        weight_(weight), scale_inv_(scale_inv), epsilon_(epsilon),
        num_tokens_(num_tokens), hidden_size_(hidden_size),
        s_variance_(s_variance) {}

  void operator() [[sycl::reqd_sub_group_size(32)]](
      const sycl::nd_item<3>& item) const {
    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();
    const int wg_id       = item.get_group(2);
    const int local_id    = item.get_local_id(2);
    const int local_range = item.get_local_range(2);

    const scalar_t* input_row = input_ + wg_id * input_stride_;

    // ---- Pass 1: variance via vectorize_read_with_alignment<VEC_P1=4> ---
    // VEC_P1 is fixed to 4, identical to rms_norm_kernel in layernorm.cpp.
    // This ensures the same per-thread partial sums → same reduce_over_group
    // result → same variance as the reference kernel.
    float variance = 0.0f;

    auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_P1>& vec) {
#pragma unroll
      for (int i = 0; i < VEC_P1; ++i) {
        float x = static_cast<float>(vec.val[i]);
        variance += x * x;
      }
    };
    auto scalar_op = [&variance](const scalar_t& val) {
      float x = static_cast<float>(val);
      variance += x * x;
    };

    vllm::vectorize_read_with_alignment<VEC_P1>(
        input_row, hidden_size_, local_id, local_range, vec_op, scalar_op);

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: vectorized normalize + weight + fp8 (VEC_P2) -----------
    // VEC_P2 = gcd(16/sizeof, hidden_size): maximise 128-bit loads.
    // No reduction → width does not affect numerical results.
    const float s_variance_val = *s_variance_ptr;
    const int   num_vec        = hidden_size_ / VEC_P2;

    using vec_t = vec_n_t<scalar_t, VEC_P2>;
    const auto* v_in = reinterpret_cast<const vec_t*>(input_row);
    const auto* v_w  = reinterpret_cast<const vec_t*>(weight_);

    for (int idx = local_id; idx < num_vec; idx += local_range) {
      vec_t src1 = v_in[idx];
      vec_t src2 = v_w[idx];
#pragma unroll
      for (int j = 0; j < VEC_P2; ++j) {
        float x = static_cast<float>(src1.val[j]);
        float out_norm = static_cast<float>(
            static_cast<scalar_t>(x * s_variance_val) * src2.val[j]);
        out_[wg_id * hidden_size_ + idx * VEC_P2 + j] =
            fp8::scaled_fp8_conversion<true, fp8_type>(out_norm, scale_inv_);
      }
    }
  }

 private:
  fp8_type* __restrict__ out_;
  const scalar_t* __restrict__ input_;
  const int input_stride_;
  const scalar_t* __restrict__ weight_;
  const float scale_inv_;
  const float epsilon_;
  const int num_tokens_;
  const int hidden_size_;
  sycl::local_accessor<float, 1> s_variance_;
};

// ===================================================================
// Kernel 2: fused_add_rms_norm_static_fp8_quant
//
// VEC_P1: Pass 1 VEC_SIZE — fixed to 1 (scalar), matching
//         fused_add_rms_norm_kernel in layernorm.cpp which uses a pure
//         scalar loop (one element per thread per iteration).
// VEC_P2: Pass 2 VEC_SIZE — gcd(16/sizeof, hidden_size).
// ===================================================================
template <typename scalar_t, int VEC_P1, int VEC_P2, typename fp8_type>
class fused_add_rms_norm_static_fp8_quant_kernel {
 public:
  fused_add_rms_norm_static_fp8_quant_kernel(
      fp8_type* __restrict__ out,
      scalar_t* __restrict__ input,
      const int input_stride,
      scalar_t* __restrict__ residual,
      const scalar_t* __restrict__ weight,
      const float scale_inv,
      const float epsilon,
      const int num_tokens,
      const int hidden_size,
      sycl::local_accessor<float, 1> s_variance)
      : out_(out), input_(input), input_stride_(input_stride),
        residual_(residual), weight_(weight), scale_inv_(scale_inv),
        epsilon_(epsilon), num_tokens_(num_tokens), hidden_size_(hidden_size),
        s_variance_(s_variance) {}

  void operator() [[sycl::reqd_sub_group_size(32)]](
      const sycl::nd_item<3>& item) const {
    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();
    const int wg_id       = item.get_group(2);
    const int local_id    = item.get_local_id(2);
    const int local_range = item.get_local_range(2);

    const scalar_t* input_row    = input_    + wg_id * input_stride_;
    scalar_t*       residual_row = residual_ + wg_id * hidden_size_;

    // ---- Pass 1: fused residual-add + variance (VEC_P1=4) ----------------
    float variance = 0.0f;

    auto vec_op = [&variance](vec_n_t<scalar_t, VEC_P1>& res_vec,
                              const vec_n_t<scalar_t, VEC_P1>& in_vec) {
      res_vec += in_vec;
#pragma unroll
      for (int i = 0; i < VEC_P1; ++i) {
        float x = static_cast<float>(res_vec.val[i]);
        variance += x * x;
      }
    };
    auto scalar_op = [&variance](scalar_t& res_elem, const scalar_t& in_elem) {
      res_elem  += in_elem;
      float x    = static_cast<float>(res_elem);
      variance  += x * x;
    };

    vllm::vectorize_fused_add_with_alignment<VEC_P1>(
        input_row, residual_row, hidden_size_,
        local_id, local_range, vec_op, scalar_op);

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: normalize + weight + fp8 (VEC_P2) -----------------------
    const float s_variance_val = *s_variance_ptr;
    const int   num_vec        = hidden_size_ / VEC_P2;

    using vec_t = vec_n_t<scalar_t, VEC_P2>;
    const auto* v_res = reinterpret_cast<const vec_t*>(residual_row);
    const auto* v_w   = reinterpret_cast<const vec_t*>(weight_);

    for (int idx = local_id; idx < num_vec; idx += local_range) {
      vec_t src1 = v_res[idx];
      vec_t src2 = v_w[idx];
#pragma unroll
      for (int j = 0; j < VEC_P2; ++j) {
        float x = static_cast<float>(src1.val[j]);
        float out_norm = static_cast<float>(
            static_cast<scalar_t>(x * s_variance_val) * src2.val[j]);
        out_[wg_id * hidden_size_ + idx * VEC_P2 + j] =
            fp8::scaled_fp8_conversion<true, fp8_type>(out_norm, scale_inv_);
      }
    }
  }

 private:
  fp8_type* __restrict__ out_;
  scalar_t* __restrict__ input_;
  const int input_stride_;
  scalar_t* __restrict__ residual_;
  const scalar_t* __restrict__ weight_;
  const float scale_inv_;
  const float epsilon_;
  const int num_tokens_;
  const int hidden_size_;
  sycl::local_accessor<float, 1> s_variance_;
};

}  // namespace vllm


// ===================================================================
// Host-side entry points
// ===================================================================

namespace vllm {

// Pass 1 VEC_SIZE for rms_norm: fixed to 4, matching rms_norm_kernel in
// layernorm.cpp (VEC_SIZE=4 vec4 loop).
static constexpr int RMS_NORM_VEC_P1 = 4;

// Pass 1 VEC_SIZE for fused_add_rms_norm: fixed to 1 (scalar), matching
// fused_add_rms_norm_kernel in layernorm.cpp which is a pure scalar loop.
static constexpr int FUSED_ADD_VEC_P1 = 1;

template <typename scalar_t, typename fp8_t>
void call_rms_norm_static_fp8_quant_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float scale_inv,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size  = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens   = input.numel() / hidden_size;

  sycl::range<3> num_work_groups(1, 1, num_tokens);
  int wg = std::min(hidden_size, 1024);
  wg = ((wg + 31) / 32) * 32;
  sycl::range<3> work_group_size(1, 1, wg);
  auto& queue = vllm::xpu::vllmGetQueue();

  const int vec_p2 =
      std::gcd(static_cast<int>(16 / sizeof(sycl_t)), hidden_size);

  VLLM_DISPATCH_VEC_SIZE(vec_p2, [&] {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(num_work_groups * work_group_size,
                            work_group_size),
          vllm::rms_norm_static_fp8_quant_kernel<
              sycl_t, RMS_NORM_VEC_P1, vec_size, fp8_t>(
              out.data_ptr<fp8_t>(),
              reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>()),
              input_stride,
              reinterpret_cast<const sycl_t*>(weight.data_ptr<scalar_t>()),
              scale_inv, epsilon,
              num_tokens, hidden_size, s_variance));
    });
  });
}

template <typename scalar_t, typename fp8_t>
void call_fused_add_rms_norm_static_fp8_quant_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float scale_inv,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size  = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens   = input.numel() / hidden_size;

  sycl::range<3> num_work_groups(1, 1, num_tokens);
  int wg = std::min(hidden_size, 1024);
  wg = ((wg + 31) / 32) * 32;
  sycl::range<3> work_group_size(1, 1, wg);
  auto& queue = vllm::xpu::vllmGetQueue();

  const int vec_p2 =
      std::gcd(static_cast<int>(16 / sizeof(sycl_t)), hidden_size);

  VLLM_DISPATCH_VEC_SIZE(vec_p2, [&] {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(num_work_groups * work_group_size,
                            work_group_size),
          vllm::fused_add_rms_norm_static_fp8_quant_kernel<
              sycl_t, FUSED_ADD_VEC_P1, vec_size, fp8_t>(
              out.data_ptr<fp8_t>(),
              reinterpret_cast<sycl_t*>(input.data_ptr<scalar_t>()),
              input_stride,
              reinterpret_cast<sycl_t*>(residual.data_ptr<scalar_t>()),
              reinterpret_cast<const sycl_t*>(weight.data_ptr<scalar_t>()),
              scale_inv, epsilon,
              num_tokens, hidden_size, s_variance));
    });
  });
}

}  // namespace vllm

void rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  const float scale_inv = 1.0f / scale.item<float>();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "rms_norm_kernel_fp8_type", [&] {
              vllm::call_rms_norm_static_fp8_quant_kernel<scalar_t, fp8_t>(
                  out, input, weight, scale_inv,
                  static_cast<float>(epsilon));
            });
      });
}

void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(residual.scalar_type() == input.scalar_type());
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  const float scale_inv = 1.0f / scale.item<float>();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fused_add_rms_norm_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "fused_add_rms_norm_kernel_fp8_type", [&] {
              vllm::call_fused_add_rms_norm_static_fp8_quant_kernel<
                  scalar_t, fp8_t>(
                  out, input, residual, weight, scale_inv,
                  static_cast<float>(epsilon));
            });
      });
}

