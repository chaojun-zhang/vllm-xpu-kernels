// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

/*
 * SYCL kernels for fused quantized RMS normalization on XPU.
 *
 * Two entry points:
 *   - rms_norm_static_fp8_quant
 *   - fused_add_rms_norm_static_fp8_quant
 *
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

    // ---- Pass 1: variance (VEC_P1=4, matches rms_norm_kernel) -----------
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

    // ---- Pass 2: normalize + weight + fp8 (VEC_P2) ----------------------
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
// Kernel 2a: fused_add_rms_norm_static_fp8_quant — vec variant
//
// width: number of scalar_t elements per packed load (8 for FP16/BF16,
//        giving 128-bit global memory ops).
//
// Pass 1 — vectorized residual-add + variance using _f16Vec<scalar_t,width>:
//   for each vec-chunk:
//     temp = input_v[stride_id]        (vectorized load from strided input)
//     temp += residual_v[id]           (_f16Vec +=, float-promoted per pair)
//     variance += temp.sum_squares()   (float32 partial sum)
//     residual_v[id] = temp            (vectorized store)
//
// Pass 2 — vectorized normalize + weight + fp8:
//   temp  = residual_v[id]
//   temp *= s_variance_val             (_f16Vec *= float)
//   temp *= weight_v[idx]             (_f16Vec *= _f16Vec)
//   per-element: fp8 = scaled_fp8_conversion(float(temp.val[i]), scale_inv)
// ===================================================================
template <typename scalar_t, int width, typename fp8_type>
class fused_add_rms_norm_static_fp8_quant_vec_kernel {
 public:
  fused_add_rms_norm_static_fp8_quant_vec_kernel(
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
    static_assert(std::is_pod_v<vec_n_t<scalar_t, width>>);
    static_assert(sizeof(vec_n_t<scalar_t, width>) ==
                  sizeof(scalar_t) * width);

    using vec_t = vec_n_t<scalar_t, width>;

    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();
    const int wg_id       = item.get_group(2);
    const int local_id    = item.get_local_id(2);
    const int local_range = item.get_local_range(2);

    const int vec_hidden_size = hidden_size_ / width;
    const int vec_input_stride = input_stride_ / width;

    auto* __restrict__ residual_v =
        reinterpret_cast<vec_t*>(residual_);
    const auto* __restrict__ weight_v =
        reinterpret_cast<const vec_t*>(weight_);

    // ---- Pass 1: fused residual-add + variance --------------------------
    // Use scalar element-by-element iteration to match the reference
    // fused_add_rms_norm_kernel in layernorm.cpp exactly.  This ensures
    // every thread accumulates the same partial variance so that
    // reduce_over_group produces a bit-exact result.
    float variance = 0.0f;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      scalar_t z = input_[wg_id * input_stride_ + idx];
      z += residual_[wg_id * hidden_size_ + idx];
      float x = static_cast<float>(z);
      variance += x * x;
      residual_[wg_id * hidden_size_ + idx] = z;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: normalize + weight + fp8 --------------------------------
    const float s_variance_val = *s_variance_ptr;
    for (int idx = local_id; idx < vec_hidden_size; idx += local_range) {
      const int id = wg_id * vec_hidden_size + idx;
      vec_t temp = residual_v[id];
      temp *= s_variance_val;
      temp *= weight_v[idx];
#pragma unroll
      for (int i = 0; i < width; ++i) {
        out_[id * width + i] =
            fp8::scaled_fp8_conversion<true, fp8_type>(
                static_cast<float>(temp.val[i]), scale_inv_);
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

// ===================================================================
// Kernel 2b: fused_add_rms_norm_static_fp8_quant — scalar fallback
//
// Pure scalar loop — no vectorization, no packed ops.
//
// Pass 1 is structurally identical to fused_add_rms_norm_kernel in
// layernorm.cpp (same loop, same fp16 add, same fp32 variance accum),
// guaranteeing bit-exact variance / rsqrt with that reference kernel
// when the same work-group size is used.
//
// Pass 2 uses the same per-element precision path as the vec kernel:
//   fp8 = scaled_fp8_conversion(float(fp16(residual*rsqrt) * weight), scale_inv)
// ===================================================================
template <typename scalar_t, typename fp8_type>
class fused_add_rms_norm_static_fp8_quant_scalar_kernel {
 public:
  fused_add_rms_norm_static_fp8_quant_scalar_kernel(
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

    // ---- Pass 1: fused residual-add + variance (scalar) -----------------
    // Mirrors fused_add_rms_norm_kernel in layernorm.cpp exactly:
    //   z  = input[row*input_stride + idx]   (may be strided)
    //   z += residual[row*hidden + idx]      (fp16 hardware add)
    //   variance += float(z)^2
    //   residual[row*hidden + idx] = z
    float variance = 0.0f;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      scalar_t z = input_[wg_id * input_stride_ + idx];
      z += residual_[wg_id * hidden_size_ + idx];
      float x = static_cast<float>(z);
      variance += x * x;
      residual_[wg_id * hidden_size_ + idx] = z;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: normalize + weight + fp8 --------------------------------
    const float s_variance_val = *s_variance_ptr;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x = static_cast<float>(residual_[wg_id * hidden_size_ + idx]);
      float out_norm = static_cast<float>(
          static_cast<scalar_t>(x * s_variance_val) * weight_[idx]);
      out_[wg_id * hidden_size_ + idx] =
          fp8::scaled_fp8_conversion<true, fp8_type>(out_norm, scale_inv_);
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

// Vectorized launch width for fused_add: 8 × fp16/bf16 = 128-bit load.
static constexpr int FUSED_ADD_VEC_WIDTH = 8;

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
  // Work-group size matches fused_add_rms_norm_kernel in layernorm.cpp:
  // std::min(hidden_size, 1024) rounded up to reqd_sub_group_size (32).
  // No num_tokens branching — keeps wg consistent with the reference so
  // reduce_over_group fp32 accumulation order is identical.
  int wg = std::min(hidden_size, 1024);
  wg = ((wg + 31) / 32) * 32;
  sycl::range<3> work_group_size(1, 1, wg);
  auto& queue = vllm::xpu::vllmGetQueue();

  // Use the vec kernel when all pointers are 16-byte aligned and hidden_size /
  // input_stride are multiples of FUSED_ADD_VEC_WIDTH — mirrors the upstream
  // CUDA dispatch condition.
  const auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  const auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  const auto wt_ptr  = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  const bool ptrs_aligned = (inp_ptr % 16 == 0) &&
                            (res_ptr % 16 == 0) &&
                            (wt_ptr  % 16 == 0);
  const bool dims_aligned = (hidden_size  % FUSED_ADD_VEC_WIDTH == 0) &&
                            (input_stride % FUSED_ADD_VEC_WIDTH == 0);


  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    if (ptrs_aligned && dims_aligned) {
      cgh.parallel_for(
          sycl::nd_range<3>(num_work_groups * work_group_size,
                            work_group_size),
          vllm::fused_add_rms_norm_static_fp8_quant_vec_kernel<
              sycl_t, FUSED_ADD_VEC_WIDTH, fp8_t>(
              out.data_ptr<fp8_t>(),
              reinterpret_cast<sycl_t*>(input.data_ptr<scalar_t>()),
              input_stride,
              reinterpret_cast<sycl_t*>(residual.data_ptr<scalar_t>()),
              reinterpret_cast<const sycl_t*>(weight.data_ptr<scalar_t>()),
              scale_inv, epsilon,
              num_tokens, hidden_size, s_variance));
    } else {
      cgh.parallel_for(
          sycl::nd_range<3>(num_work_groups * work_group_size,
                            work_group_size),
          vllm::fused_add_rms_norm_static_fp8_quant_scalar_kernel<
              sycl_t, fp8_t>(
              out.data_ptr<fp8_t>(),
              reinterpret_cast<sycl_t*>(input.data_ptr<scalar_t>()),
              input_stride,
              reinterpret_cast<sycl_t*>(residual.data_ptr<scalar_t>()),
              reinterpret_cast<const sycl_t*>(weight.data_ptr<scalar_t>()),
              scale_inv, epsilon,
              num_tokens, hidden_size, s_variance));
    }
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