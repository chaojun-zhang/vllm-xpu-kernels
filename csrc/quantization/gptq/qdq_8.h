/*
 * Adapted from https://github.com/turboderp/exllamav2
 * SYCL/XPU port of qdq_8.cuh
 */

#pragma once
#include <cstdint>

namespace vllm {
namespace gptq {

// 8-bit shuffle is a no-op (already in sequential order).
inline void shuffle_8bit_4(uint32_t* q, int stride) {}

}  // namespace gptq
}  // namespace vllm
