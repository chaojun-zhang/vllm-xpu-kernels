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
//   sum_squares()    — float-accumulated Σ float(val[i])²
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

// ---------------------------------------------------------------------------
// vectorize_read_with_alignment
//
// SYCL port of vllm/csrc/quantization/vectorization_utils.cuh
// (read-only variant).
//
// Iterates over `in[0..len)` with vectorized loads where alignment allows,
// falling back to scalar for the unaligned prefix and any trailing tail.
//
//   vec_op(const vec_n_t<InT,VEC_SIZE>&)  — called for each aligned vector
//   scalar_op(const InT&)                 — called for each scalar element
//
// The interface is identical to the CUDA version so kernels ported from
// layernorm_quant_kernels.cu can be written the same way.
// ---------------------------------------------------------------------------
template <int VEC_SIZE, typename InT, typename VecOp, typename ScaOp>
inline void vectorize_read_with_alignment(const InT* in, int len,
                                          int tid, int stride,
                                          VecOp&& vec_op,
                                          ScaOp&& scalar_op) {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * static_cast<int>(sizeof(InT));
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  // fast path: whole region is aligned and len is a multiple of VEC_SIZE
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    const int num_vec = len / VEC_SIZE;
    const auto* v_in = reinterpret_cast<const vec_n_t<InT, VEC_SIZE>*>(in);
    for (int i = tid; i < num_vec; i += stride) {
      vec_op(v_in[i]);
    }
    return;
  }

  // slow path: handle unaligned prefix, aligned body, and tail
  int misalignment_offset = static_cast<int>(addr & (WIDTH - 1));
  int alignment_bytes     = WIDTH - misalignment_offset;
  int prefix_elems        = alignment_bytes & (WIDTH - 1);
  prefix_elems /= static_cast<int>(sizeof(InT));
  prefix_elems = prefix_elems < len ? prefix_elems : len;

  // 1. unaligned prefix — scalar
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(in[i]);
  }

  in  += prefix_elems;
  len -= prefix_elems;

  // 2. aligned body — vectorized
  const int num_vec = len / VEC_SIZE;
  const auto* v_in = reinterpret_cast<const vec_n_t<InT, VEC_SIZE>*>(in);
  for (int i = tid; i < num_vec; i += stride) {
    vec_op(v_in[i]);
  }

  // 3. tail — scalar
  const int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(in[i]);
  }
}

// ---------------------------------------------------------------------------
// vectorize_fused_add_with_alignment
//
// Read-write variant for fused_add_rms_norm Pass 1.
//
// Iterates over two arrays in parallel — `in` (may be strided, read-only)
// and `inout` (contiguous, read+write) — with vectorized access where
// alignment allows, falling back to scalar otherwise.
//
// The alignment decision is made on `inout` (contiguous residual), which
// is the stricter constraint; `in` is only ever read so partial alignment
// there is safe with scalar fallback.
//
//   vec_op(vec_n_t<T,N>& inout_vec, const vec_n_t<T,N>& in_vec)
//       — called for each aligned vector pair; inout_vec is written back
//   scalar_op(T& inout_elem, const T& in_elem)
//       — called for each scalar element pair; inout_elem is written back
// ---------------------------------------------------------------------------
template <int VEC_SIZE, typename T, typename VecOp, typename ScaOp>
inline void vectorize_fused_add_with_alignment(
    const T* in,       // read-only, may be strided
    T* inout,          // read+write, contiguous
    int len,           // number of elements (hidden_size)
    int tid, int stride,
    VecOp&&  vec_op,
    ScaOp&&  scalar_op) {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * static_cast<int>(sizeof(T));
  uintptr_t addr = reinterpret_cast<uintptr_t>(inout);

  // fast path: inout is aligned and len is a multiple of VEC_SIZE
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    const int num_vec = len / VEC_SIZE;
    const auto* v_in    = reinterpret_cast<const vec_n_t<T, VEC_SIZE>*>(in);
    auto*       v_inout = reinterpret_cast<vec_n_t<T, VEC_SIZE>*>(inout);
    for (int i = tid; i < num_vec; i += stride) {
      vec_op(v_inout[i], v_in[i]);
    }
    return;
  }

  // slow path: scalar prefix + aligned vec body + scalar tail
  int misalignment_offset = static_cast<int>(addr & (WIDTH - 1));
  int alignment_bytes     = WIDTH - misalignment_offset;
  int prefix_elems        = alignment_bytes & (WIDTH - 1);
  prefix_elems /= static_cast<int>(sizeof(T));
  prefix_elems = prefix_elems < len ? prefix_elems : len;

  // 1. unaligned prefix — scalar
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(inout[i], in[i]);
  }

  in    += prefix_elems;
  inout += prefix_elems;
  len   -= prefix_elems;

  // 2. aligned body — vectorized
  const int num_vec = len / VEC_SIZE;
  const auto* v_in    = reinterpret_cast<const vec_n_t<T, VEC_SIZE>*>(in);
  auto*       v_inout = reinterpret_cast<vec_n_t<T, VEC_SIZE>*>(inout);
  for (int i = tid; i < num_vec; i += stride) {
    vec_op(v_inout[i], v_in[i]);
  }

  // 3. tail — scalar
  const int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(inout[i], in[i]);
  }
}

}  // namespace vllm
