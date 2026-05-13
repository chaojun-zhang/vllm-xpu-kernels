/*
 * Adapted from https://github.com/turboderp/exllamav2
 * SYCL/XPU port of qdq_2.cuh
 */

#pragma once
#include <cstdint>

namespace vllm {
namespace gptq {

// Permutation:
//
// ffddbb99 77553311  eeccaa88 66442200

inline void shuffle_2bit_16(uint32_t* q, int stride) {
  uint32_t qa = q[0];
  uint32_t qb = 0;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint32_t qa0 = qa & 0x03;
    uint32_t qa1 = (qa & 0x0c) >> 2;
    qa >>= 4;
    qb |= (qa1 << (i * 2 + 16));
    qb |= (qa0 << (i * 2));
  }
  q[0] = qb;
}

}  // namespace gptq
}  // namespace vllm
