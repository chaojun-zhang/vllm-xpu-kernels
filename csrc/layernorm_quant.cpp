// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

/*
 * SYCL kernels for fused quantized RMS normalization on XPU.
 * Corresponds to the CUDA kernels in vllm/csrc/layernorm_quant_kernels.cu.
 *
 * Two entry points:
 *   - rms_norm_static_fp8_quant:
 *       out[fp8] = fp8( rms_norm(input) * weight / scale )
 *
 *   - fused_add_rms_norm_static_fp8_quant:
 *       residual += input          (in-place update)
 *       out[fp8] = fp8( rms_norm(residual) * weight / scale )
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
#include "type_convert.h"
#include "utils.h"

namespace vllm {

// ===================================================================
// Kernel 1a: rms_norm_static_fp8_quant  (vectorized, VEC_SIZE > 1)
//
// Mirrors the CUDA rms_norm_static_fp8_quant_kernel<scalar_t, fp8_type, VEC_SIZE>.
// Uses aligned vec_n_t<scalar_t, VEC_SIZE> loads/stores.
//
// out[fp8]  = fp8( (input / rms) * weight / scale )
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
      : out_(out),
        input_(input),
        input_stride_(input_stride),
        weight_(weight),
        scale_inv_(scale_inv),
        epsilon_(epsilon),
        num_tokens_(num_tokens),
        hidden_size_(hidden_size),
        s_variance_(s_variance) {}

  void operator() [[sycl::reqd_sub_group_size(32)]](
      const sycl::nd_item<3>& item) const {
    // vec_n_t carries float-promoted operators (*= float, *= vec, sum_squares)
    // that exactly match _f16Vec semantics — fp16/bf16 arithmetic is always
    // done as: fp16(float(a) op float(b)).  This ensures bit-exact agreement
    // with the reference path: rms_norm (uses _f16Vec) → static_fp8_quant.
    using vec_t = vllm::vec_n_t<scalar_t, VEC_SIZE>;

    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();

    const int wg_id = item.get_group(2);
    const int local_id = item.get_local_id(2);
    const int local_range = item.get_local_range(2);
    const int num_vec = hidden_size_ / VEC_SIZE;
    const int vec_input_stride = input_stride_ / VEC_SIZE;

    // no device-side printf — launch params are printed on the host

    const auto* __restrict__ input_v =
        reinterpret_cast<const vec_t*>(input_);
    const auto* __restrict__ weight_v =
        reinterpret_cast<const vec_t*>(weight_);

    // ---- Pass 1: compute variance = sum(x^2) ----------------------------
    // sum_squares() uses float-promoted accumulation, same as _f16Vec.
    float variance = 0.0f;

    for (int idx = local_id; idx < num_vec; idx += local_range) {
      vec_t temp = input_v[wg_id * vec_input_stride + idx];
      variance += temp.sum_squares();
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());

    if (local_id == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: normalize, weight, quantise to fp8 ---------------------
    // temp *= rsqrt  → each element: fp16(float(x) * rsqrt)
    // temp *= weight → each element: fp16(float(normed) * float(w))
    // Then fp8 = scaled_fp8_conversion(float(fp16_out), scale_inv)
    // This is identical in structure to fused_add_rms_norm_static_fp8_quant_vec_kernel.
    const float s_variance_val = *s_variance_ptr;

    for (int idx = local_id; idx < num_vec; idx += local_range) {
      int id = wg_id * num_vec + idx;
      vec_t temp = input_v[wg_id * vec_input_stride + idx];
      temp *= s_variance_val;       // fp16(float(x) * rsqrt)   — like _f16Vec *= float
      temp *= weight_v[idx];        // fp16(float(n) * float(w)) — like _f16Vec *= vec
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        out_[id * VEC_SIZE + i] =
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
// Kernel 1b: rms_norm_static_fp8_quant  (scalar, VEC_SIZE == 1)
//
// Scalar fallback — no vectorization.
// Mirrors the CUDA (width == 0) specialization.
//
// out[fp8]  = fp8( (input / rms) * weight / scale )
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
      : out_(out),
        input_(input),
        input_stride_(input_stride),
        weight_(weight),
        scale_inv_(scale_inv),
        epsilon_(epsilon),
        num_tokens_(num_tokens),
        hidden_size_(hidden_size),
        s_variance_(s_variance) {}

  void operator() [[sycl::reqd_sub_group_size(32)]](
      const sycl::nd_item<3>& item) const {
    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();

    const int wg_id = item.get_group(2);
    const int local_id = item.get_local_id(2);
    const int local_range = item.get_local_range(2);

    // ---- Pass 1: compute variance = sum(x^2) ----------------------------
    float variance = 0.0f;

    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x = static_cast<float>(input_[wg_id * input_stride_ + idx]);
      variance += x * x;
    }

    // Work-group reduction
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());

    if (local_id == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: normalize, weight, quantise to fp8 ---------------------
    const float s_variance_val = *s_variance_ptr;

    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x = static_cast<float>(input_[wg_id * input_stride_ + idx]);
      // Use float-promoted multiply (matching _f16Vec::operator*= semantics)
      // to exactly match the reference rms_norm + static_scaled_fp8_quant path.
      scalar_t normed   = static_cast<scalar_t>(x * s_variance_val);
      float    out_norm = static_cast<float>(normed) *
                          static_cast<float>(weight_[idx]);
      out_[wg_id * hidden_size_ + idx] =
          fp8::scaled_fp8_conversion<true, fp8_type>(
              static_cast<float>(static_cast<scalar_t>(out_norm)), scale_inv_);
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
// Kernel 2a: fused_add_rms_norm_static_fp8_quant  (vectorized, width > 0)
//
// Specialization for FP16/BF16 with packed + vectorized ops.
// Mirrors the CUDA (width > 0) specialization.
//
// residual += input              (in-place)
// out[fp8]  = fp8( (residual / rms) * weight / scale )
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
      : out_(out),
        input_(input),
        input_stride_(input_stride),
        residual_(residual),
        weight_(weight),
        scale_inv_(scale_inv),
        epsilon_(epsilon),
        num_tokens_(num_tokens),
        hidden_size_(hidden_size),
        s_variance_(s_variance) {}

  void operator() [[sycl::reqd_sub_group_size(32)]](
      const sycl::nd_item<3>& item) const {
    // Sanity checks on our vector struct and type-punned pointer arithmetic
    static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
    static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

    using vec_t = _f16Vec<scalar_t, width>;

    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();

    const int wg_id = item.get_group(2);
    const int local_id = item.get_local_id(2);
    const int local_range = item.get_local_range(2);
    const int vec_hidden_size = hidden_size_ / width;
    const int vec_input_stride = input_stride_ / width;

    /* These and the argument pointers are all declared `restrict` as they are
       not aliased in practice. Argument pointers should not be dereferenced
       in this kernel as that would be undefined behavior */
    auto* __restrict__ input_v = reinterpret_cast<vec_t*>(input_);
    auto* __restrict__ residual_v = reinterpret_cast<vec_t*>(residual_);
    auto const* __restrict__ weight_v =
        reinterpret_cast<const vec_t*>(weight_);

    // ---- Pass 1: residual += input, compute variance --------------------
    float variance = 0.0f;

    for (int idx = local_id; idx < vec_hidden_size; idx += local_range) {
      int stride_id = wg_id * vec_input_stride + idx;
      int id = wg_id * vec_hidden_size + idx;
      vec_t temp = input_v[stride_id];
      temp += residual_v[id];
      variance += temp.sum_squares();
      residual_v[id] = temp;
    }

    // Work-group reduction
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());

    if (local_id == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: normalize, weight, quantise to fp8 ---------------------
    // invert scale to avoid division
    const float s_variance_val = *s_variance_ptr;

    for (int idx = local_id; idx < vec_hidden_size; idx += local_range) {
      int id = wg_id * vec_hidden_size + idx;
      vec_t temp = residual_v[id];
      temp *= s_variance_val;
      temp *= weight_v[idx];
      for (int i = 0; i < width; ++i) {
        out_[id * width + i] =
            fp8::scaled_fp8_conversion<true, fp8_type>(
                static_cast<float>(temp.data[i]), scale_inv_);
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
// Kernel 2b: fused_add_rms_norm_static_fp8_quant  (generic, width == 0)
//
// Scalar fallback — no vectorization, no packed ops.
// Mirrors the CUDA (width == 0) specialization.
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
      : out_(out),
        input_(input),
        input_stride_(input_stride),
        residual_(residual),
        weight_(weight),
        scale_inv_(scale_inv),
        epsilon_(epsilon),
        num_tokens_(num_tokens),
        hidden_size_(hidden_size),
        s_variance_(s_variance) {}

  void operator() [[sycl::reqd_sub_group_size(32)]](
      const sycl::nd_item<3>& item) const {
    float* s_variance_ptr =
        s_variance_.template get_multi_ptr<sycl::access::decorated::no>().get();

    const int wg_id = item.get_group(2);
    const int local_id = item.get_local_id(2);
    const int local_range = item.get_local_range(2);

    // ---- Pass 1: residual += input, compute variance --------------------
    float variance = 0.0f;
    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      scalar_t z = input_[wg_id * input_stride_ + idx];
      z += residual_[wg_id * hidden_size_ + idx];
      float x = static_cast<float>(z);
      variance += x * x;
      residual_[wg_id * hidden_size_ + idx] = z;
    }

    // Work-group reduction
    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance, sycl::plus<>());

    if (local_id == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size_ + epsilon_);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ---- Pass 2: normalize, weight, quantise to fp8 ---------------------
    // invert scale to avoid division
    const float s_variance_val = *s_variance_ptr;

    for (int idx = local_id; idx < hidden_size_; idx += local_range) {
      float x = static_cast<float>(residual_[wg_id * hidden_size_ + idx]);
      float const out_norm = static_cast<float>(
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

// Dispatches scalar_t and fp8_t, then launches the appropriate kernel
// (vectorized or scalar) based on the vec_size_val template parameter.
// ===================================================================
#define LAUNCH_RMS_NORM(vec_size_val)                                          \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "rms_norm_static_fp8_quant_scalar_type", [&] {      \
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;      \
        VLLM_DISPATCH_FP8_TYPES(                                               \
            out.scalar_type(), "rms_norm_static_fp8_quant_fp8_type", [&] {     \
              queue.submit([&](sycl::handler& cgh) {                           \
                sycl::local_accessor<float, 1> s_variance(                     \
                    sycl::range<1>(1), cgh);                                   \
                if constexpr ((vec_size_val) > 1) {                            \
                  cgh.parallel_for(                                            \
                      sycl::nd_range<3>(                                       \
                          num_work_groups * work_group_size,                    \
                          work_group_size),                                     \
                      vllm::rms_norm_static_fp8_quant_vec_kernel<              \
                          sycl_t, (vec_size_val), fp8_t>(                      \
                          out.data_ptr<fp8_t>(),                               \
                          reinterpret_cast<const sycl_t*>(                     \
                              input.data_ptr<scalar_t>()),                     \
                          input_stride,                                        \
                          reinterpret_cast<const sycl_t*>(                     \
                              weight.data_ptr<scalar_t>()),                    \
                          scale_inv,                                           \
                          static_cast<float>(epsilon),                         \
                          num_tokens,                                          \
                          hidden_size,                                         \
                          s_variance));                                        \
                } else {                                                       \
                  cgh.parallel_for(                                            \
                      sycl::nd_range<3>(                                       \
                          num_work_groups * work_group_size,                    \
                          work_group_size),                                     \
                      vllm::rms_norm_static_fp8_quant_scalar_kernel<           \
                          sycl_t, fp8_t>(                                      \
                          out.data_ptr<fp8_t>(),                               \
                          reinterpret_cast<const sycl_t*>(                     \
                              input.data_ptr<scalar_t>()),                     \
                          input_stride,                                        \
                          reinterpret_cast<const sycl_t*>(                     \
                              weight.data_ptr<scalar_t>()),                    \
                          scale_inv,                                           \
                          static_cast<float>(epsilon),                         \
                          num_tokens,                                          \
                          hidden_size,                                         \
                          s_variance));                                        \
                }                                                              \
              });                                                              \
            });                                                                \
      });

void rms_norm_static_fp8_quant(
    torch::Tensor& out,     // [..., hidden_size]  fp8
    torch::Tensor& input,   // [..., hidden_size]
    torch::Tensor& weight,  // [hidden_size]
    torch::Tensor& scale,   // [1]
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  // Pre-compute inverted scale on host to avoid per-work-item global read.
  const float scale_inv = 1.0f / scale.item<float>();

  sycl::range<3> num_work_groups(1, 1, num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller work-group size allows
     for increased EU occupancy and better latency
     hiding on global mem ops. */
  const int max_work_group_size = (num_tokens < 256) ? 1024 : 256;
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_static_fp8_quant_scalar_type_outer", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const int calculated_vec_size =
            std::gcd((int)(16 / sizeof(sycl_t)), hidden_size);
        const int block_size =
            std::min(hidden_size / calculated_vec_size, max_work_group_size);
        int wg = ((block_size + 31) / 32) * 32;
        sycl::range<3> work_group_size(1, 1, wg);

        VLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
          constexpr int vec_size_val = vec_size;
          LAUNCH_RMS_NORM(vec_size_val);
        });
      });
}

// ===================================================================
// LAUNCH_FUSED_ADD_RMS_NORM macro
// ===================================================================
#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                       \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "fused_add_rms_norm_kernel_scalar_type", [&] {      \
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;      \
        VLLM_DISPATCH_FP8_TYPES(                                               \
            out.scalar_type(), "fused_add_rms_norm_kernel_fp8_type", [&] {     \
              queue.submit([&](sycl::handler& cgh) {                           \
                sycl::local_accessor<float, 1> s_variance(                     \
                    sycl::range<1>(1), cgh);                                   \
                if constexpr ((width) > 0) {                                   \
                  cgh.parallel_for(                                            \
                      sycl::nd_range<3>(                                       \
                          num_work_groups * work_group_size,                    \
                          work_group_size),                                     \
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
                          scale_inv,                                           \
                          static_cast<float>(epsilon),                         \
                          num_tokens,                                          \
                          hidden_size,                                         \
                          s_variance));                                        \
                } else {                                                       \
                  cgh.parallel_for(                                            \
                      sycl::nd_range<3>(                                       \
                          num_work_groups * work_group_size,                    \
                          work_group_size),                                     \
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
                          scale_inv,                                           \
                          static_cast<float>(epsilon),                         \
                          num_tokens,                                          \
                          hidden_size,                                         \
                          s_variance));                                        \
                }                                                              \
              });                                                              \
            });                                                                \
      });

void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,       // [..., hidden_size]  fp8
    torch::Tensor& input,     // [..., hidden_size]
    torch::Tensor& residual,  // [..., hidden_size]
    torch::Tensor& weight,    // [hidden_size]
    torch::Tensor& scale,     // [1]
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(residual.scalar_type() == input.scalar_type());
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  const float scale_inv = 1.0f / scale.item<float>();

  sycl::range<3> num_work_groups(1, 1, num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller work-group size allows
     for increased EU occupancy and better latency
     hiding on global mem ops. */
  const int max_work_group_size = (num_tokens < 256) ? 1024 : 256;
  int wg = std::min(hidden_size, max_work_group_size);
  wg = ((wg + 31) / 32) * 32;
  sycl::range<3> work_group_size(1, 1, wg);
  auto& queue = vllm::xpu::vllmGetQueue();

  /* If the tensor types are FP16/BF16, try to use the optimized kernel
     with packed + vectorized ops.
     Max optimization is achieved with a width-8 vector of FP16/BF16s
     since we can load at most 128 bits at once in a global memory op.
     However, this requires each tensor's data to be aligned to 16
     bytes. */
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

