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
 * The test compares each fused kernel against the unfused reference:
 *   rms_norm / fused_add_rms_norm  (layernorm.cpp)  +  static_scaled_fp8_quant
 *
 * To get bit-exact agreement we must replicate the reference exactly:
 *
 * 1. Work-group size: layernorm.cpp always uses wg = min(hidden_size, 1024).
 *    A different wg changes the number of partial sums entering
 *    reduce_over_group, which changes the float32 summation order (fp32
 *    addition is not associative) → different variance → different rsqrt →
 *    occasionally different fp16-normalised value → fp8 boundary crossing.
 *
 * 2. Pass 1 must be scalar (one element per loop iteration), matching the
 *    reference.  Grouping elements into wider vectors changes the per-thread
 *    partial sum, causing the same summation-order problem even if the wg
 *    size is matched.
 *
 * 3. Pass 2 uses hardware fp16/bf16 multiply for the weight step, matching
 *    the reference:  out = ((scalar_t)(float(x)*rsqrt)) * weight[i]
 *    (not float-promoted: float(normed)*float(weight) rounds differently).
 *
 * Pass 2 is safe to vectorize because it has no reduction — it is purely
 * elementwise.  We use vec_n_t<scalar_t, VEC_SIZE> for 128-bit loads there.
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
// Kernel 1a: rms_norm_static_fp8_quant  (vectorized pass 2)
// Pass 1: scalar — identical to rms_norm_kernel in layernorm.cpp
// Pass 2: vec_n_t<scalar_t,VEC_SIZE> loads, hw fp16 multiply
// ===================================================================
template <typename scalar_t, int VEC_SIZE, typename fp8_type>
class rms_norm_static_fp8_quant_vec_kernel {
 public:
  rms_norm_static_fp8_quant_vec_kernel(
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
    using vec_t = vllm::vec_n_t<scalar_t, VEC_SIZE>;

    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();
    const int wg_id       = item.get_group(2);
    const int local_id    = item.get_local_id(2);
    const int local_range = item.get_local_range(2);

    // ---- Pass 1: scalar variance — identical to rms_norm_kernel ----------
    float variance = 0.0f;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x = static_cast<float>(input_[wg_id * input_stride_ + idx]);
      variance += x * x;
    }
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: vectorized normalize + weight + fp8 ---------------------
    // Matches reference pass 2: ((scalar_t)(float(x)*rsqrt)) * weight[j]
    //   vec *= rsqrt  →  val[i] = (scalar_t)(float(val[i]) * rsqrt)
    //   vec *= weight →  val[i] = val[i] * weight.val[i]  (hw fp16 *)
    const float s_variance_val = *s_variance_ptr;
    const int   num_vec         = hidden_size_ / VEC_SIZE;
    const int   vec_stride      = input_stride_ / VEC_SIZE;

    const auto* __restrict__ input_v  = reinterpret_cast<const vec_t*>(input_);
    const auto* __restrict__ weight_v = reinterpret_cast<const vec_t*>(weight_);

    for (int idx = local_id; idx < num_vec; idx += local_range) {
      vec_t temp = input_v[wg_id * vec_stride + idx];
      temp *= s_variance_val;          // (scalar_t)(float(x)*rsqrt)
      temp *= weight_v[idx];           // hw fp16: normed * weight
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        out_[(wg_id * num_vec + idx) * VEC_SIZE + i] =
            fp8::scaled_fp8_conversion<true, fp8_type>(
                static_cast<float>(temp.val[i]), scale_inv_);
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
// Kernel 1b: rms_norm_static_fp8_quant  (scalar fallback)
// ===================================================================
template <typename scalar_t, typename fp8_type>
class rms_norm_static_fp8_quant_scalar_kernel {
 public:
  rms_norm_static_fp8_quant_scalar_kernel(
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

    // ---- Pass 1: scalar variance -----------------------------------------
    float variance = 0.0f;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x = static_cast<float>(input_[wg_id * input_stride_ + idx]);
      variance += x * x;
    }
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: scalar normalize + weight + fp8 -------------------------
    // ((scalar_t)(float(x)*rsqrt)) * weight[idx]  — hw fp16 multiply
    const float s_variance_val = *s_variance_ptr;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x     = static_cast<float>(input_[wg_id * input_stride_ + idx]);
      scalar_t normed   = static_cast<scalar_t>(x * s_variance_val);
      scalar_t out_fp16 = normed * weight_[idx];   // hw fp16 *
      out_[wg_id * hidden_size_ + idx] =
          fp8::scaled_fp8_conversion<true, fp8_type>(
              static_cast<float>(out_fp16), scale_inv_);
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
    const int wg_id       = item.get_group(2);
    const int local_id    = item.get_local_id(2);
    const int local_range = item.get_local_range(2);

    // ---- Pass 1: scalar residual-add + variance --------------------------
    // Identical to fused_add_rms_norm_kernel:
    //   z = (scalar_t)input[stride*i] + residual[hidden*i]  (hw fp16 +)
    //   variance += float(z)^2
    //   residual[hidden*i] = z
    float variance = 0.0f;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      scalar_t z = input_[wg_id * input_stride_ + idx];
      z += residual_[wg_id * hidden_size_ + idx];
      variance += static_cast<float>(z) * static_cast<float>(z);
      residual_[wg_id * hidden_size_ + idx] = z;
    }
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: vectorized normalize + weight + fp8 ---------------------
    const float s_variance_val  = *s_variance_ptr;
    const int   vec_hidden_size = hidden_size_ / width;

    auto* __restrict__       residual_v = reinterpret_cast<vec_t*>(residual_);
    const auto* __restrict__ weight_v   = reinterpret_cast<const vec_t*>(weight_);

    for (int idx = local_id; idx < vec_hidden_size; idx += local_range) {
      int id   = wg_id * vec_hidden_size + idx;
      vec_t temp = residual_v[id];
      temp *= s_variance_val;   // (scalar_t)(float(x)*rsqrt)
      temp *= weight_v[idx];    // hw fp16: normed * weight
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
    float variance = 0.0f;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      scalar_t z = input_[wg_id * input_stride_ + idx];
      z += residual_[wg_id * hidden_size_ + idx];
      variance += static_cast<float>(z) * static_cast<float>(z);
      residual_[wg_id * hidden_size_ + idx] = z;
    }
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());
    if (local_id == 0)
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: scalar normalize + weight + fp8 -------------------------
    const float s_variance_val = *s_variance_ptr;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x     = static_cast<float>(residual_[wg_id * hidden_size_ + idx]);
      scalar_t normed   = static_cast<scalar_t>(x * s_variance_val);
      scalar_t out_fp16 = normed * weight_[idx];   // hw fp16 *
      out_[wg_id * hidden_size_ + idx] =
          fp8::scaled_fp8_conversion<true, fp8_type>(
              static_cast<float>(out_fp16), scale_inv_);
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

#define LAUNCH_RMS_NORM(vec_size_val)                                          \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "rms_norm_static_fp8_quant_scalar_type", [&] {      \
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;      \
        VLLM_DISPATCH_FP8_TYPES(                                               \
            out.scalar_type(), "rms_norm_static_fp8_quant_fp8_type", [&] {     \
              if constexpr ((vec_size_val) > 1) {                              \
                fprintf(stderr,                                                \
                    "[rms_norm_static_fp8_quant] vec_kernel"                   \
                    " num_tokens=%d hidden_size=%d input_stride=%d"            \
                    " VEC_SIZE=%d wg=%zu\n",                                   \
                    num_tokens, hidden_size, input_stride,                     \
                    (vec_size_val), work_group_size[2]);                       \
              } else {                                                         \
                fprintf(stderr,                                                \
                    "[rms_norm_static_fp8_quant] scalar_kernel"                \
                    " num_tokens=%d hidden_size=%d input_stride=%d"            \
                    " wg=%zu\n",                                               \
                    num_tokens, hidden_size, input_stride,                     \
                    work_group_size[2]);                                       \
              }                                                                \
              queue.submit([&](sycl::handler& cgh) {                          \
                sycl::local_accessor<float, 1> s_variance(                     \
                    sycl::range<1>(1), cgh);                                   \
                if constexpr ((vec_size_val) > 1) {                            \
                  cgh.parallel_for(                                            \
                      sycl::nd_range<3>(num_work_groups * work_group_size,     \
                                        work_group_size),                      \
                      vllm::rms_norm_static_fp8_quant_vec_kernel<              \
                          sycl_t, (vec_size_val), fp8_t>(                      \
                          out.data_ptr<fp8_t>(),                               \
                          reinterpret_cast<const sycl_t*>(                     \
                              input.data_ptr<scalar_t>()),                     \
                          input_stride,                                        \
                          reinterpret_cast<const sycl_t*>(                     \
                              weight.data_ptr<scalar_t>()),                    \
                          scale_inv, static_cast<float>(epsilon),              \
                          num_tokens, hidden_size, s_variance));               \
                } else {                                                       \
                  cgh.parallel_for(                                            \
                      sycl::nd_range<3>(num_work_groups * work_group_size,     \
                                        work_group_size),                      \
                      vllm::rms_norm_static_fp8_quant_scalar_kernel<           \
                          sycl_t, fp8_t>(                                      \
                          out.data_ptr<fp8_t>(),                               \
                          reinterpret_cast<const sycl_t*>(                     \
                              input.data_ptr<scalar_t>()),                     \
                          input_stride,                                        \
                          reinterpret_cast<const sycl_t*>(                     \
                              weight.data_ptr<scalar_t>()),                    \
                          scale_inv, static_cast<float>(epsilon),              \
                          num_tokens, hidden_size, s_variance));               \
                }                                                              \
              });                                                              \
            });                                                                \
      });

void rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& scale,
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  int hidden_size  = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens   = input.numel() / hidden_size;

  const float scale_inv = 1.0f / scale.item<float>();

  sycl::range<3> num_work_groups(1, 1, num_tokens);

  // Match rms_norm_kernel: wg = min(hidden_size, 1024) rounded to 32.
  // Critical for bit-exact variance: same wg → same partial-sum count for
  // reduce_over_group → same float32 summation order → identical rsqrt.
  int wg = std::min(hidden_size, 1024);
  wg = ((wg + 31) / 32) * 32;
  sycl::range<3> work_group_size(1, 1, wg);
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_static_fp8_quant_scalar_type_outer", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

        // Choose vec_size for pass-2 vectorization only.
        // All pointers must be suitably aligned and hidden_size divisible.
        auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
        auto wt_ptr  = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
        auto out_ptr = reinterpret_cast<std::uintptr_t>(out.data_ptr());
        const int max_vec = static_cast<int>(16 / sizeof(sycl_t));
        const int calculated_vec_size = std::gcd(max_vec, hidden_size);
        const bool ptrs_aligned =
            inp_ptr % (calculated_vec_size * sizeof(sycl_t)) == 0 &&
            wt_ptr  % (calculated_vec_size * sizeof(sycl_t)) == 0 &&
            out_ptr % (calculated_vec_size * sizeof(sycl_t)) == 0 &&
            input_stride % calculated_vec_size == 0;
        const int vec_size_actual = ptrs_aligned ? calculated_vec_size : 1;

        VLLM_DISPATCH_VEC_SIZE(vec_size_actual, [&] {
          constexpr int vec_size_val = vec_size;
          LAUNCH_RMS_NORM(vec_size_val);
        });
      });
}

// ===================================================================
#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                       \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "fused_add_rms_norm_kernel_scalar_type", [&] {      \
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;      \
        VLLM_DISPATCH_FP8_TYPES(                                               \
            out.scalar_type(), "fused_add_rms_norm_kernel_fp8_type", [&] {     \
              if constexpr ((width) > 0) {                                     \
                fprintf(stderr,                                                \
                    "[fused_add_rms_norm_static_fp8_quant] vec_kernel"         \
                    " num_tokens=%d hidden_size=%d input_stride=%d"            \
                    " width=%d wg=%zu\n",                                      \
                    num_tokens, hidden_size, input_stride,                     \
                    (width), work_group_size[2]);                              \
              } else {                                                         \
                fprintf(stderr,                                                \
                    "[fused_add_rms_norm_static_fp8_quant] scalar_kernel"      \
                    " num_tokens=%d hidden_size=%d input_stride=%d"            \
                    " wg=%zu\n",                                               \
                    num_tokens, hidden_size, input_stride,                     \
                    work_group_size[2]);                                       \
              }                                                                \
              queue.submit([&](sycl::handler& cgh) {                          \
                sycl::local_accessor<float, 1> s_variance(                     \
                    sycl::range<1>(1), cgh);                                   \
                if constexpr ((width) > 0) {                                   \
                  cgh.parallel_for(                                            \
                      sycl::nd_range<3>(num_work_groups * work_group_size,     \
                                        work_group_size),                      \
                      vllm::fused_add_rms_norm_static_fp8_quant_vec_kernel<    \
                          sycl_t, (width), fp8_t>(                             \
                          out.data_ptr<fp8_t>(),                               \
                          reinterpret_cast<sycl_t*>(                           \
                              input.data_ptr<scalar_t>()),                     \
                          input_stride,                                        \
                          reinterpret_cast<sycl_t*>(                           \
                              residual.data_ptr<scalar_t>()),                  \
                          reinterpret_cast<const sycl_t*>(                     \
                              weight.data_ptr<scalar_t>()),                    \
                          scale_inv, static_cast<float>(epsilon),              \
                          num_tokens, hidden_size, s_variance));               \
                } else {                                                       \
                  cgh.parallel_for(                                            \
                      sycl::nd_range<3>(num_work_groups * work_group_size,     \
                                        work_group_size),                      \
                      vllm::fused_add_rms_norm_static_fp8_quant_scalar_kernel< \
                          sycl_t, fp8_t>(                                      \
                          out.data_ptr<fp8_t>(),                               \
                          reinterpret_cast<sycl_t*>(                           \
                              input.data_ptr<scalar_t>()),                     \
                          input_stride,                                        \
                          reinterpret_cast<sycl_t*>(                           \
                              residual.data_ptr<scalar_t>()),                  \
                          reinterpret_cast<const sycl_t*>(                     \
                              weight.data_ptr<scalar_t>()),                    \
                          scale_inv, static_cast<float>(epsilon),              \
                          num_tokens, hidden_size, s_variance));               \
                }                                                              \
              });                                                              \
            });                                                                \
      });

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
  int hidden_size  = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens   = input.numel() / hidden_size;

  const float scale_inv = 1.0f / scale.item<float>();

  sycl::range<3> num_work_groups(1, 1, num_tokens);

  // Match fused_add_rms_norm_kernel: wg = min(hidden_size, 1024) rounded to 32.
  int wg = std::min(hidden_size, 1024);
  wg = ((wg + 31) / 32) * 32;
  sycl::range<3> work_group_size(1, 1, wg);
  auto& queue = vllm::xpu::vllmGetQueue();

  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr  = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0 && input_stride % 8 == 0) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

