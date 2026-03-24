// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once
#include <cstdlib>
#include <string>

namespace vllm {

// Returns true if the environment variable VLLM_BATCH_INVARIANT is set to 1.
// Mirrors the upstream CUDA implementation in csrc/core/batch_invariant.hpp.
inline bool vllm_is_batch_invariant() {
  static bool cached = []() {
    const char* val = std::getenv("VLLM_BATCH_INVARIANT");
    return (val && std::atoi(val) != 0) ? true : false;
  }();
  return cached;
}

}  // namespace vllm

