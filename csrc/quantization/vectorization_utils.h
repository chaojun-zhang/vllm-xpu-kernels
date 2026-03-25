// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "vectorization.h"

namespace vllm {
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
}  // namespace vllm
