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
 * fused_add_rms_norm_static_fp8_quant — unchanged (vec + scalar kernels kept
 * as-is, matching the CUDA fused_add upstream).
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
#include "utils.h"

namespace vllm {

// ===================================================================
// Kernel 1: rms_norm_static_fp8_quant  (unified vec + scalar)
//
// Mirrors CUDA: rms_norm_static_fp8_quant_kernel<scalar_t, fp8_type, VEC_SIZE>
//
// Pass 1: vllm::vectorize_read_with_alignment<VEC_SIZE>
//         — handles unaligned prefix / aligned body / tail,
//           identical structure to the CUDA upstream.
//         — VEC_SIZE=1 ⇒ pure scalar (no vec path entered).
// Pass 2: direct vec loop over hidden_size/VEC_SIZE elements,
//         mirrors CUDA pass 2 (v_in / v_w indexed by idx).
// ===================================================================
// TODO: Further optimize this kernel.
template <typename scalar_t, int VEC_SIZE, typename fp8_type>
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

    // ---- Pass 1: variance via vectorize_read_with_alignment --------------
    // Identical structure to CUDA upstream:
    //   vec_op  : variance += vec.sum_squares()
    //   scalar_op: variance += x*x
    // Handles unaligned prefix, aligned body, and tail automatically.
    float variance = 0.0f;

    auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
      variance += vec.sum_squares();
    };
    auto scalar_op = [&variance](const scalar_t& val) {
      float x = static_cast<float>(val);
      variance += x * x;
    };

    vllm::vectorize_read_with_alignment<VEC_SIZE>(
        input_row, hidden_size_, local_id, local_range, vec_op, scalar_op);

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: vectorized normalize + weight + fp8 ---------------------
    // Mirrors CUDA pass 2: v_in/v_w indexed by idx (0..hidden/VEC_SIZE).
    // No alignment handling needed here — output is always contiguous.
    const float s_variance_val = *s_variance_ptr;
    const int   num_vec        = hidden_size_ / VEC_SIZE;

    using vec_t = vec_n_t<scalar_t, VEC_SIZE>;
    const auto* v_in = reinterpret_cast<const vec_t*>(input_row);
    const auto* v_w  = reinterpret_cast<const vec_t*>(weight_);

    for (int idx = local_id; idx < num_vec; idx += local_range) {
      vec_t src1 = v_in[idx];
      vec_t src2 = v_w[idx];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float x = static_cast<float>(src1.val[j]);
        // ((scalar_t)(x * rsqrt)) * weight[j]  — hw scalar_t multiply
        float out_norm = static_cast<float>(
            static_cast<scalar_t>(x * s_variance_val) * src2.val[j]);
        out_[wg_id * hidden_size_ + idx * VEC_SIZE + j] =
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
// Kernel 2a: fused_add_rms_norm_static_fp8_quant  (vectorized pass 2)
// Pass 1: scalar — identical to fused_add_rms_norm_kernel in layernorm.cpp
// Pass 2: vec_n_t<scalar_t,width> loads, hw fp16 multiply
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
    static_assert(sizeof(vec_n_t<scalar_t, width>) == sizeof(scalar_t) * width);
    using vec_t = vec_n_t<scalar_t, width>;

    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();
    const int wg_id          = item.get_group(2);
    const int local_id       = item.get_local_id(2);
    const int local_range    = item.get_local_range(2);
    const int vec_hidden_size = hidden_size_ / width;
    const int vec_input_stride = input_stride_ / width;

    auto* __restrict__       input_v    = reinterpret_cast<vec_t*>(input_);
    auto* __restrict__       residual_v = reinterpret_cast<vec_t*>(residual_);
    const auto* __restrict__ weight_v   = reinterpret_cast<const vec_t*>(weight_);

    // ---- Pass 1: vectorized residual-add + variance ----------------------
    // Mirrors CUDA upstream vec kernel exactly:
    //   stride_id = wg_id * vec_input_stride + idx  (input may be strided)
    //   id        = wg_id * vec_hidden_size  + idx  (residual is contiguous)
    //   temp = input_v[stride_id] + residual_v[id]
    //   variance += temp.sum_squares()
    //   residual_v[id] = temp
    float variance = 0.0f;
    for (int idx = local_id; idx < vec_hidden_size; idx += local_range) {
      int stride_id = wg_id * vec_input_stride + idx;
      int id        = wg_id * vec_hidden_size  + idx;
      vec_t temp    = input_v[stride_id];
      temp         += residual_v[id];
      variance     += temp.sum_squares();
      residual_v[id] = temp;
    }
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: vectorized normalize + weight + fp8 ---------------------
    // Mirrors CUDA upstream vec kernel pass 2.
    const float s_variance_val = *s_variance_ptr;
    for (int idx = local_id; idx < vec_hidden_size; idx += local_range) {
      int id     = wg_id * vec_hidden_size + idx;
      vec_t temp = residual_v[id];
      temp *= s_variance_val;   // (scalar_t)(float(x)*rsqrt)
      temp *= weight_v[idx];    // hw scalar_t: normed * weight
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
// Kernel 2b: fused_add_rms_norm_static_fp8_quant  (scalar fallback)
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

    // ---- Pass 1: scalar residual-add + variance --------------------------
    // Mirrors CUDA upstream scalar path exactly:
    //   z = input[wg_id * input_stride + idx] + residual[wg_id * hidden + idx]
    //   variance += (float)z * (float)z
    //   residual[wg_id * hidden + idx] = z
    float variance = 0.0f;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      scalar_t z = input_[wg_id * input_stride_ + idx];
      z += residual_[wg_id * hidden_size_ + idx];
      float x = (float)z;
      variance += x * x;
      residual_[wg_id * hidden_size_ + idx] = z;
    }
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: scalar normalize + weight + fp8 -------------------------
    // Mirrors CUDA upstream scalar path:
    //   out_norm = ((scalar_t)(x * s_variance)) * weight[idx]
    const float s_variance_val = *s_variance_ptr;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x = (float)residual_[wg_id * hidden_size_ + idx];
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

  const int calculated_vec_size =
      std::gcd(static_cast<int>(16 / sizeof(sycl_t)), hidden_size);

  VLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(num_work_groups * work_group_size,
                            work_group_size),
          vllm::rms_norm_static_fp8_quant_kernel<sycl_t, vec_size, fp8_t>(
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

  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr  = reinterpret_cast<std::uintptr_t>(weight.data_ptr());


  auto vec_launch = [&]() {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(num_work_groups * work_group_size,
                            work_group_size),
          vllm::fused_add_rms_norm_static_fp8_quant_vec_kernel<
              sycl_t, 8, fp8_t>(
              out.data_ptr<fp8_t>(),
              reinterpret_cast<sycl_t*>(input.data_ptr<scalar_t>()),
              input_stride,
              reinterpret_cast<sycl_t*>(residual.data_ptr<scalar_t>()),
              reinterpret_cast<const sycl_t*>(weight.data_ptr<scalar_t>()),
              scale_inv, epsilon,
              num_tokens, hidden_size, s_variance));

    });
  };
  auto scalar_launch = [&]() {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
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
    });
  };

  bool ptrs_are_aligned =
        inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0 && input_stride % 8 == 0) {
    vec_launch();
  } else {
    scalar_launch();
  }
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

