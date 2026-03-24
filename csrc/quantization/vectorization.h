// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

namespace vllm {

// Vectorization containers.
//
// vec_n_t<scalar_t, N>: aligned vector struct with float-promoted operators
// mirroring _f16Vec semantics.  alignas(sizeof(scalar_t)*N) is valid because
// both sizeof and N are compile-time constants.
//
// q8_n_t<quant_t, N>: aligned container for quantized (fp8/int8) values.

template <typename scalar_t, size_t vec_size>
struct alignas(sizeof(scalar_t) * vec_size) vec_n_t {
  static_assert(vec_size > 0 && (vec_size & (vec_size - 1)) == 0,
                "vec_size must be a positive power of 2");
  scalar_t val[vec_size];

  // Float-promoted multiply by scalar (matches _f16Vec::operator*=(float))
  vec_n_t& operator*=(const float scale) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      val[i] = static_cast<scalar_t>(static_cast<float>(val[i]) * scale);
    }
    return *this;
  }

  // Float-promoted elementwise multiply (matches _f16Vec::operator*=(vec))
  vec_n_t& operator*=(const vec_n_t& other) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      val[i] = static_cast<scalar_t>(
          static_cast<float>(val[i]) * static_cast<float>(other.val[i]));
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
