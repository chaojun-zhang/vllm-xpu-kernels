// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

namespace vllm {

// Vectorization containers.
//
// vec_n_t<scalar_t, N>: aligned vector struct whose arithmetic operators
// mirror the reference layernorm kernels in layernorm.cpp:
//
//   operator+=(vec)  — float-promoted add: val[i] = fp16(float(a)+float(b))
//   operator*=(float)— float*scalar then fp16: val[i] = fp16(float(val)*scale)
//   operator*=(vec)  — hardware scalar_t multiply: val[i] = val[i] * other.val[i]
//
// The mixed precision in *=(float) vs *=(vec) reflects the reference pass 2:
//   dst = ((scalar_t)(float(x) * rsqrt)) * weight   ← step1: float*rsqrt→fp16
//                                                    ← step2: fp16 * fp16 hw
//
// q8_n_t<quant_t, N>: aligned container for quantized (fp8/int8) values.

template <typename scalar_t, size_t vec_size>
struct alignas(sizeof(scalar_t) * vec_size) vec_n_t {
  static_assert(vec_size > 0 && (vec_size & (vec_size - 1)) == 0,
                "vec_size must be a positive power of 2");
  scalar_t val[vec_size];

  vec_n_t& operator+=(const vec_n_t& other) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      val[i] = val[i] + other.val[i];  // hardware scalar_t add
    }
    return *this;
  }

  vec_n_t& operator*=(const float scale) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      val[i] = static_cast<scalar_t>(static_cast<float>(val[i]) * scale);
    }
    return *this;
  }

  vec_n_t& operator*=(const vec_n_t& other) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      val[i] = val[i] * other.val[i];  // hardware scalar_t multiply
    }
    return *this;
  }

  // Sum of squares over all elements
  float sum_squares() const {
    float result = 0.0f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      float x = static_cast<float>(val[i]);
      result += x * x;
    }
    return result;
  }
};

template <typename quant_type_t, size_t vec_size>
struct alignas(sizeof(quant_type_t) * vec_size) q8_n_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                std::is_same_v<quant_type_t, c10::Float8_e5m2>);
  quant_type_t val[vec_size];
};

template <typename scalar_t>
using vec4_t = vec_n_t<scalar_t, 4>;
template <typename quant_type_t>
using q8x4_t = q8_n_t<quant_type_t, 4>;

}  // namespace vllm
