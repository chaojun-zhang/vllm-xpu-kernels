// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>

namespace vllm {

/* Vector POD struct to generate vectorized and packed FP16/BF16 ops
   for appropriate specializations of fused_add_rms_norm_kernel and
   fused_add_rms_norm_static_fp8_quant_kernel.
   Only functions that are necessary in those kernels are implemented.
   Alignment to 16 bytes is required to use 128-bit global memory ops.

   This is the SYCL equivalent of the CUDA _f16Vec in type_convert.cuh.
   Unlike the CUDA version which relies on _typeConvert and HIP/CUDA packed
   intrinsics, the SYCL version performs all arithmetic through float
   conversions since sycl::half and sycl::ext::oneapi::bfloat16 support
   direct static_cast<float>.
 */
template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  /* Not theoretically necessary that width is a power of 2 but should
     almost always be the case for optimization purposes */
  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  scalar_t data[width];

  _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float a0 = static_cast<float>(data[i]);
        float a1 = static_cast<float>(data[i + 1]);
        float b0 = static_cast<float>(other.data[i]);
        float b1 = static_cast<float>(other.data[i + 1]);
        data[i] = static_cast<scalar_t>(a0 + b0);
        data[i + 1] = static_cast<scalar_t>(a1 + b1);
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        data[i] = static_cast<scalar_t>(
            static_cast<float>(data[i]) + static_cast<float>(other.data[i]));
      }
    }
    return *this;
  }

  _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float a0 = static_cast<float>(data[i]);
        float a1 = static_cast<float>(data[i + 1]);
        float b0 = static_cast<float>(other.data[i]);
        float b1 = static_cast<float>(other.data[i + 1]);
        data[i] = static_cast<scalar_t>(a0 * b0);
        data[i + 1] = static_cast<scalar_t>(a1 * b1);
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        data[i] = static_cast<scalar_t>(
            static_cast<float>(data[i]) * static_cast<float>(other.data[i]));
      }
    }
    return *this;
  }

  _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        data[i] = static_cast<scalar_t>(static_cast<float>(data[i]) * scale);
        data[i + 1] =
            static_cast<scalar_t>(static_cast<float>(data[i + 1]) * scale);
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        data[i] = static_cast<scalar_t>(static_cast<float>(data[i]) * scale);
      }
    }
    return *this;
  }

  float sum_squares() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float x0 = static_cast<float>(data[i]);
        float x1 = static_cast<float>(data[i + 1]);
        result += x0 * x0 + x1 * x1;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = static_cast<float>(data[i]);
        result += x * x;
      }
    }
    return result;
  }
};

}  // namespace vllm

