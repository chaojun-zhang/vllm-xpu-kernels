#include <sycl/sycl.hpp>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"

#define VLLM_LDG(arg) *(arg)

namespace vllm {

template <typename T>
inline T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template <typename T>
inline T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
inline T gelu_new_kernel(const T& x) {
  // 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
inline T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + (T)sycl::exp(-1.702f * (float)x)));
}

template <typename T>
inline T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + sycl::erf(f * ALPHA)));
}

template <typename T>
inline T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + sycl::tanh(inner)));
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
inline scalar_t compute(const scalar_t& x, const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
class act_kernel {
 public:
  act_kernel(scalar_t* __restrict__ out,          // [..., d]
             const scalar_t* __restrict__ input,  // [..., d]
             const int d)
      : out_(out), input_(input), d_(d) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    const int64_t token_idx = item_ct1.get_group(2);
    for (int64_t idx = item_ct1.get_local_id(2); idx < d_;
         idx += item_ct1.get_local_range(2)) {
      const scalar_t x = input_[token_idx * d_ + idx];
      out_[token_idx * d_ + idx] = ACT_FN(x);
    }
  }

 private:
  scalar_t* __restrict__ out_;          // [..., d]
  const scalar_t* __restrict__ input_;  // [..., d]
  const int d_;
};

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
class act_and_mul_kernel {
 public:
  act_and_mul_kernel(scalar_t* __restrict__ out,          // [..., d]
                     const scalar_t* __restrict__ input,  // [..., 2, d]
                     const int d)
      : out_(out), input_(input), d_(d) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    const int64_t token_idx = item_ct1.get_group(2);
    for (int64_t idx = item_ct1.get_local_id(2); idx < d_;
         idx += item_ct1.get_local_range(2)) {
      const scalar_t x = input_[token_idx * 2 * d_ + idx];
      const scalar_t y = input_[token_idx * 2 * d_ + d_ + idx];
      out_[token_idx * d_ + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
  }

 private:
  scalar_t* __restrict__ out_;          // [..., d]
  const scalar_t* __restrict__ input_;  // [..., 2, d]
  const int d_;
};

template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) T
swigluoai_and_mul(const T& gate, const T& up, float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + sycl::exp(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
class swigluoai_and_mul_kernel {
 public:
  swigluoai_and_mul_kernel(scalar_t* __restrict__ out,          // [..., d]
                           const scalar_t* __restrict__ input,  // [..., 2, d]
                           const int d, const float alpha, const float limit)
      : out(out), input(input), d(d), alpha(alpha), limit(limit) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    for (int64_t idx = item.get_local_id(0); idx < d;
         idx += item.get_local_range(0)) {
      // gate = x[..., ::2]  (even indices)
      const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
      // up = x[..., 1::2]   (odd indices)
      const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);

      out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
    }
  }

 private:
  scalar_t* out;          // [..., d]
  const scalar_t* input;  // [..., topk, d]
  const int d;
  const float alpha;
  const float limit;
};

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                  \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;                \
  int d = input.size(-1) / 2;                                             \
  int64_t num_tokens = input.numel() / input.size(-1);                    \
  sycl::range<3> grid(1, 1, num_tokens);                                  \
  sycl::range<3> block(1, 1, std::min(d, 1024));                          \
  if (num_tokens == 0) {                                                  \
    return;                                                               \
  }                                                                       \
  auto out_ptr = out.data_ptr<scalar_t>();                                \
  auto input_ptr = input.data_ptr<scalar_t>();                            \
  at::DeviceGuard device_guard(input.device());                           \
  auto& queue = vllm::xpu::vllmGetQueue();                                \
  queue.submit([&](sycl::handler& cgh) {                                  \
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block),              \
                     vllm::act_and_mul_kernel<sycl_t, KERNEL, ACT_FIRST>( \
                         (sycl_t*)out_ptr, (sycl_t*)input_ptr, d));       \
  });

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
  });
}

void mul_and_silu(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mul_and_silu", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
  });
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
  });
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_tanh_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
  });
}

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;                     \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  sycl::range<3> grid(1, 1, num_tokens);                                       \
  sycl::range<3> block(1, 1, std::min(d, 1024));                               \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  auto out_ptr = out.data_ptr<scalar_t>();                                     \
  auto input_ptr = input.data_ptr<scalar_t>();                                 \
  at::DeviceGuard device_guard(input.device());                                \
  auto& queue = vllm::xpu::vllmGetQueue();                                     \
  queue.submit([&](sycl::handler& cgh) {                                       \
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                   \
                     vllm::act_kernel<sycl_t, KERNEL>((sycl_t*)out_ptr,        \
                                                      (sycl_t*)input_ptr, d)); \
  });

#define LAUNCH_SWIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                     \
  int d = input.size(-1) / 2;                                              \
  int64_t num_tokens = input.numel() / input.size(-1);                     \
  sycl::range<1> grid(num_tokens);                                         \
  sycl::range<1> block(std::min(d, 1024));                                 \
  at::DeviceGuard device_guard(input.device());                            \
  auto& queue = vllm::xpu::vllmGetQueue();                                 \
  VLLM_DISPATCH_FLOATING_TYPES(                                            \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {        \
        queue.submit([&](sycl::handler& cgh) {                             \
          cgh.parallel_for(                                                \
              sycl::nd_range<1>(grid * block, block),                      \
              vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>(  \
                  out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d, \
                  ALPHA, LIMIT));                                          \
        });                                                                \
      });

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_new", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
  });
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_fast", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
  });
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_quick", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
  });
}

void swigluoai_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SWIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}

// ---------------------------------------------------------------------------
// silu_and_mul_per_block_quant
// Fused SiLU(gate) * up → group-level dynamic quantization.
//
// Layout: input is [num_tokens, hidden_size * 2] (gate || up concatenated).
// One SYCL work-group handles one (token, column-group) pair.
// Work-group size == group_size so that every work-item covers exactly one
// element, and sycl::reduce_over_group gives the whole-group maximum.
// ---------------------------------------------------------------------------

namespace vllm {

template <typename scalar_t, typename out_t>
class silu_and_mul_per_block_quant_kernel {
 public:
  silu_and_mul_per_block_quant_kernel(
      out_t* __restrict__ out_,
      float* __restrict__ scales_,
      const scalar_t* __restrict__ input_,
      const float* scale_ub_,
      const int hidden_size_,
      const int num_groups_,
      const int group_size_,
      const int64_t scale_stride_token_,
      const int64_t scale_stride_group_)
      : out(out_),
        scales(scales_),
        input(input_),
        scale_ub(scale_ub_),
        hidden_size(hidden_size_),
        num_groups(num_groups_),
        group_size(group_size_),
        scale_stride_token(scale_stride_token_),
        scale_stride_group(scale_stride_group_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int wg_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));

    // Map flat work-group id → (token_idx, group_idx)
    const int token_idx = wg_id / num_groups;
    const int group_idx = wg_id % num_groups;
    const int group_start = group_idx * group_size;

    // Input: [gate || up] along last dim, stride = hidden_size * 2
    const scalar_t* token_gate =
        input + token_idx * hidden_size * 2 + group_start;
    const scalar_t* token_up = token_gate + hidden_size;
    out_t* token_output = out + token_idx * hidden_size + group_start;

    // Step 1: SiLU(gate[tid]) * up[tid]
    const float gate_f = static_cast<float>(token_gate[tid]);
    const float up_f = static_cast<float>(token_up[tid]);
    const float silu_gate = gate_f / (1.0f + sycl::exp(-gate_f));
    const float result = silu_gate * up_f;

    // Step 2: group-reduce to find max |result|
    const float group_max = sycl::reduce_over_group(
        item.get_group(), sycl::fabs(result), sycl::maximum<float>());

    // Step 3: compute scale (same value in all work-items); tid==0 writes
    float group_scale;
    if constexpr (std::is_same_v<out_t, int8_t>) {
      group_scale = (group_max > 0.0f) ? (group_max / 127.0f) : 1.0f;
    } else {
      // FP8 path
      const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);
      group_scale =
          sycl::max(group_max / fp8_max, fp8::min_scaling_factor<out_t>::val());
      if (scale_ub != nullptr) {
        group_scale = sycl::min(group_scale, *scale_ub);
        // Re-clamp after applying upper bound so scale stays >= min
        group_scale = sycl::max(
            group_scale, fp8::min_scaling_factor<out_t>::val());
      }
    }

    if (tid == 0) {
      const int64_t scale_idx =
          static_cast<int64_t>(token_idx) * scale_stride_token +
          static_cast<int64_t>(group_idx) * scale_stride_group;
      scales[scale_idx] = group_scale;
    }

    // Step 4: quantize and write
    const float inv_scale = 1.0f / group_scale;
    const float q = result * inv_scale;
    if constexpr (std::is_same_v<out_t, int8_t>) {
      token_output[tid] = static_cast<int8_t>(
          sycl::max(sycl::min(sycl::rint(q), 127.0f), -128.0f));
    } else {
      const float fp8_max = static_cast<float>(fp8::quant_type_max_v<out_t>);
      token_output[tid] =
          static_cast<out_t>(sycl::max(sycl::min(q, fp8_max), -fp8_max));
    }
  }

 private:
  out_t* __restrict__ out;
  float* __restrict__ scales;
  const scalar_t* __restrict__ input;
  const float* scale_ub;
  const int hidden_size;
  const int num_groups;
  const int group_size;
  const int64_t scale_stride_token;
  const int64_t scale_stride_group;
};

}  // namespace vllm

void silu_and_mul_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size,
    std::optional<torch::Tensor> scale_ub,
    bool is_scale_transposed) {
  TORCH_CHECK(
      out.dtype() == torch::kFloat8_e4m3fn ||
          out.dtype() == torch::kFloat8_e5m2 || out.dtype() == torch::kInt8,
      "out must be float8_e4m3fn, float8_e5m2, or int8");
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
      "input must be float16 or bfloat16");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      group_size == 64 || group_size == 128,
      "group_size must be 64 or 128, got ",
      group_size);
  if (scale_ub.has_value()) {
    TORCH_CHECK(
        out.dtype() == torch::kFloat8_e4m3fn ||
            out.dtype() == torch::kFloat8_e5m2,
        "scale_ub is only supported for FP8 output");
  }

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = out.size(-1);
  const int64_t num_groups = hidden_size / group_size;
  TORCH_CHECK(
      input.size(-1) == hidden_size * 2,
      "input last dim must be 2 * hidden_size");
  TORCH_CHECK(
      hidden_size % group_size == 0,
      "hidden_size must be divisible by group_size");

  if (num_tokens == 0) return;

  // Scale strides: mirrors rms_norm_per_block_quant convention.
  // is_scale_transposed=false → scales[num_tokens, num_groups], row-major
  // is_scale_transposed=true  → underlying [num_groups, num_tokens], col-major
  const int64_t scale_stride_token =
      is_scale_transposed ? 1LL : static_cast<int64_t>(num_groups);
  const int64_t scale_stride_group =
      is_scale_transposed ? num_tokens : 1LL;

  const at::DeviceGuard device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  const float* scale_ub_ptr =
      scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;

  // One work-group per (token, group) pair; work-group size == group_size.
  const int64_t num_wgs = num_tokens * num_groups;
  const int64_t wg_size = group_size;

  VLLM_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const auto* input_ptr =
            reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>());
        float* scales_ptr = scales.data_ptr<float>();

        auto launch = [&](auto out_ptr) {
          using out_t = std::remove_pointer_t<decltype(out_ptr)>;
          queue.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(num_wgs * wg_size, wg_size),
                vllm::silu_and_mul_per_block_quant_kernel<sycl_t, out_t>(
                    out_ptr,
                    scales_ptr,
                    input_ptr,
                    scale_ub_ptr,
                    static_cast<int>(hidden_size),
                    static_cast<int>(num_groups),
                    static_cast<int>(group_size),
                    scale_stride_token,
                    scale_stride_group));
          });
        };

        if (out.dtype() == torch::kFloat8_e4m3fn) {
          launch(out.data_ptr<at::Float8_e4m3fn>());
        } else if (out.dtype() == torch::kFloat8_e5m2) {
          launch(out.data_ptr<at::Float8_e5m2>());
        } else {
          launch(out.data_ptr<int8_t>());
        }
      });
}
