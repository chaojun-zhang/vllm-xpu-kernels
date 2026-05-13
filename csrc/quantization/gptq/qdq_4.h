/*
 * Adapted from https://github.com/turboderp/exllamav2
 * SYCL/XPU port of qdq_4.cuh
 */

#pragma once
#include <cstdint>

namespace vllm {
namespace gptq {

// Permutation:
//
// 77775555 33331111  66664444 22220000

inline void shuffle_4bit_8(uint32_t* q, int stride) {
  uint32_t qa = q[0];
  uint32_t qb = 0;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t qa0 = qa & 0x0f;
    uint32_t qa1 = (qa & 0xf0) >> 4;
    qa >>= 8;
    qb |= (qa1 << (i * 4 + 16));
    qb |= (qa0 << (i * 4));
  }
  q[0] = qb;
}

}  // namespace gptq
}  // namespace vllm
