// SPDX-License-Identifier: Apache-2.0
// Fused MLA RoPE + concat-and-cache kernel.
// Applies RoPE to q_pe (multi-head) and k_pe (single head) in-place, then
// writes k_pe and kv_c into the paged KV cache.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <string>

#include "dispatch_utils.h"
#include "quantization/fp8/quant_utils.h"
#include "utils.h"

namespace vllm {

template <typename scalar_t, bool IS_NEOX, typename cache_t,
          Fp8KVCacheDataType kv_dt>
class concat_and_cache_mla_rope_fused_kernel {
 public:
  concat_and_cache_mla_rope_fused_kernel(
      const int64_t* __restrict__ positions,
      scalar_t* __restrict__ q_pe,
      scalar_t* __restrict__ k_pe,
      const scalar_t* __restrict__ kv_c,
      const scalar_t* __restrict__ rope_cos_sin_cache,
      const int rot_dim,
      const int64_t q_pe_stride_token,
      const int64_t q_pe_stride_head,
      const int64_t k_pe_stride,
      const int64_t kv_c_stride,
      const int num_q_heads,
      cache_t* __restrict__ kv_cache,
      const int64_t* __restrict__ slot_mapping,
      const int block_stride,
      const int entry_stride,
      const int kv_lora_rank,
      const int block_size,
      const float* kv_cache_quant_scale)
      : positions_(positions),
        q_pe_(q_pe),
        k_pe_(k_pe),
        kv_c_(kv_c),
        rope_cos_sin_cache_(rope_cos_sin_cache),
        rot_dim_(rot_dim),
        q_pe_stride_token_(q_pe_stride_token),
        q_pe_stride_head_(q_pe_stride_head),
        k_pe_stride_(k_pe_stride),
        kv_c_stride_(kv_c_stride),
        num_q_heads_(num_q_heads),
        kv_cache_(kv_cache),
        slot_mapping_(slot_mapping),
        block_stride_(block_stride),
        entry_stride_(entry_stride),
        kv_lora_rank_(kv_lora_rank),
        block_size_(block_size),
        kv_cache_quant_scale_(kv_cache_quant_scale) {}

  // Each work-group handles one token.
  void operator()(const sycl::nd_item<1> item_id) const {
    const int64_t token_idx = item_id.get_group(0);
    const int64_t slot_idx = slot_mapping_[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
      return;
    }
    const int64_t pos = positions_[token_idx];

    const scalar_t* cos_sin_ptr = rope_cos_sin_cache_ + pos * rot_dim_;
    const int embed_dim = rot_dim_ / 2;

    // Q RoPE (in-place, multi-head)
    const int nq = num_q_heads_ * embed_dim;
    for (int i = item_id.get_local_id(0); i < nq;
         i += item_id.get_local_range(0)) {
      const int head_idx = i / embed_dim;
      const int pair_idx = i % embed_dim;

      scalar_t cos_val = static_cast<scalar_t>(cos_sin_ptr[pair_idx]);
      scalar_t sin_val =
          static_cast<scalar_t>(cos_sin_ptr[pair_idx + embed_dim]);

      scalar_t* q_pe_head_ptr = q_pe_ + token_idx * q_pe_stride_token_ +
                                head_idx * q_pe_stride_head_;

      int pair_idx_x, pair_idx_y;
      if constexpr (IS_NEOX) {
        // GPT-NeoX style rotary embedding.
        pair_idx_x = pair_idx;
        pair_idx_y = embed_dim + pair_idx;
      } else {
        // GPT-J style rotary embedding.
        pair_idx_x = pair_idx * 2;
        pair_idx_y = pair_idx * 2 + 1;
      }

      scalar_t x_src = q_pe_head_ptr[pair_idx_x];
      scalar_t y_src = q_pe_head_ptr[pair_idx_y];

      q_pe_head_ptr[pair_idx_x] = x_src * cos_val - y_src * sin_val;
      q_pe_head_ptr[pair_idx_y] = y_src * cos_val + x_src * sin_val;
    }

    const int64_t block_idx = slot_idx / block_size_;
    const int64_t entry_idx = slot_idx % block_size_;

    // K RoPE (single head) + write to KV cache
    for (int i = item_id.get_local_id(0); i < embed_dim;
         i += item_id.get_local_range(0)) {
      const int pair_idx = i;

      scalar_t cos_val = static_cast<scalar_t>(cos_sin_ptr[pair_idx]);
      scalar_t sin_val =
          static_cast<scalar_t>(cos_sin_ptr[pair_idx + embed_dim]);

      scalar_t* k_pe_head_ptr = k_pe_ + token_idx * k_pe_stride_;

      int pair_idx_x, pair_idx_y;
      if constexpr (IS_NEOX) {
        pair_idx_x = pair_idx;
        pair_idx_y = embed_dim + pair_idx;
      } else {
        pair_idx_x = pair_idx * 2;
        pair_idx_y = pair_idx * 2 + 1;
      }

      scalar_t x_src = k_pe_head_ptr[pair_idx_x];
      scalar_t y_src = k_pe_head_ptr[pair_idx_y];

      scalar_t x_dst = x_src * cos_val - y_src * sin_val;
      scalar_t y_dst = y_src * cos_val + x_src * sin_val;

      k_pe_head_ptr[pair_idx_x] = x_dst;
      k_pe_head_ptr[pair_idx_y] = y_dst;

      // MLA Cache Store for rotary part
      cache_t* kv_cache_ptr = kv_cache_ + block_idx * block_stride_ +
                              entry_idx * entry_stride_ + kv_lora_rank_;

      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        kv_cache_ptr[pair_idx_x] = static_cast<cache_t>(x_dst);
        kv_cache_ptr[pair_idx_y] = static_cast<cache_t>(y_dst);
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        kv_cache_ptr[pair_idx_x] = static_cast<cache_t>(
            static_cast<at::Float8_e4m3fn>(
                static_cast<float>(x_dst) * (*kv_cache_quant_scale_)));
        kv_cache_ptr[pair_idx_y] = static_cast<cache_t>(
            static_cast<at::Float8_e4m3fn>(
                static_cast<float>(y_dst) * (*kv_cache_quant_scale_)));
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
        kv_cache_ptr[pair_idx_x] = static_cast<cache_t>(
            static_cast<at::Float8_e5m2>(
                static_cast<float>(x_dst) * (*kv_cache_quant_scale_)));
        kv_cache_ptr[pair_idx_y] = static_cast<cache_t>(
            static_cast<at::Float8_e5m2>(
                static_cast<float>(y_dst) * (*kv_cache_quant_scale_)));
      }
    }

    // NoPE: copy kv_c into KV cache
    for (int i = item_id.get_local_id(0); i < kv_lora_rank_;
         i += item_id.get_local_range(0)) {
      const scalar_t src_value = kv_c_[token_idx * kv_c_stride_ + i];

      cache_t* kv_cache_ptr = kv_cache_ + block_idx * block_stride_ +
                              entry_idx * entry_stride_;

      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        kv_cache_ptr[i] = static_cast<cache_t>(src_value);
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        kv_cache_ptr[i] = static_cast<cache_t>(
            static_cast<at::Float8_e4m3fn>(
                static_cast<float>(src_value) * (*kv_cache_quant_scale_)));
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
        kv_cache_ptr[i] = static_cast<cache_t>(
            static_cast<at::Float8_e5m2>(
                static_cast<float>(src_value) * (*kv_cache_quant_scale_)));
      }
    }
  }

 private:
  const int64_t* __restrict__ positions_;
  scalar_t* __restrict__ q_pe_;
  scalar_t* __restrict__ k_pe_;
  const scalar_t* __restrict__ kv_c_;
  const scalar_t* __restrict__ rope_cos_sin_cache_;
  const int rot_dim_;
  const int64_t q_pe_stride_token_;
  const int64_t q_pe_stride_head_;
  const int64_t k_pe_stride_;
  const int64_t kv_c_stride_;
  const int num_q_heads_;
  cache_t* __restrict__ kv_cache_;
  const int64_t* __restrict__ slot_mapping_;
  const int block_stride_;
  const int entry_stride_;
  const int kv_lora_rank_;
  const int block_size_;
  const float* kv_cache_quant_scale_;
};

}  // namespace vllm

#define CALL_CONCAT_AND_CACHE_MLA_ROPE_FUSED(KV_T, CACHE_T, KV_DTYPE)       \
  if (rope_is_neox) {                                                        \
    queue.submit([&](sycl::handler& cgh) {                                   \
      cgh.parallel_for(                                                      \
          sycl::nd_range<1>(grid * block, block),                            \
          vllm::concat_and_cache_mla_rope_fused_kernel<KV_T, true, CACHE_T,  \
                                                       KV_DTYPE>(            \
              positions.data_ptr<int64_t>(),                                 \
              reinterpret_cast<KV_T*>(q_pe.data_ptr()),                      \
              reinterpret_cast<KV_T*>(k_pe.data_ptr()),                      \
              reinterpret_cast<const KV_T*>(kv_c.data_ptr()),                \
              reinterpret_cast<const KV_T*>(                                 \
                  rope_cos_sin_cache.data_ptr()),                            \
              rot_dim, q_pe_stride_token, q_pe_stride_head, k_pe_stride,     \
              kv_c_stride, num_q_heads,                                      \
              reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),               \
              slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride,  \
              kv_lora_rank, block_size,                                      \
              kv_cache_quant_scale.data_ptr<float>()));                      \
    });                                                                      \
  } else {                                                                   \
    queue.submit([&](sycl::handler& cgh) {                                   \
      cgh.parallel_for(                                                      \
          sycl::nd_range<1>(grid * block, block),                            \
          vllm::concat_and_cache_mla_rope_fused_kernel<KV_T, false, CACHE_T, \
                                                       KV_DTYPE>(            \
              positions.data_ptr<int64_t>(),                                 \
              reinterpret_cast<KV_T*>(q_pe.data_ptr()),                      \
              reinterpret_cast<KV_T*>(k_pe.data_ptr()),                      \
              reinterpret_cast<const KV_T*>(kv_c.data_ptr()),                \
              reinterpret_cast<const KV_T*>(                                 \
                  rope_cos_sin_cache.data_ptr()),                            \
              rot_dim, q_pe_stride_token, q_pe_stride_head, k_pe_stride,     \
              kv_c_stride, num_q_heads,                                      \
              reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),               \
              slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride,  \
              kv_lora_rank, block_size,                                      \
              kv_cache_quant_scale.data_ptr<float>()));                      \
    });                                                                      \
  }

// Executes RoPE on q_pe and k_pe, then writes k_pe and kv_c in the kv cache.
// q_pe and k_pe are modified in place.
// Replaces DeepseekScalingRotaryEmbedding + concat_and_cache_mla.
void concat_and_cache_mla_rope_fused(
    torch::Tensor& positions,             // [num_tokens]
    torch::Tensor& q_pe,                  // [num_tokens, num_q_heads, rot_dim]
    torch::Tensor& k_pe,                  // [num_tokens, rot_dim]
    torch::Tensor& kv_c,                  // [num_tokens, kv_lora_rank]
    torch::Tensor& rope_cos_sin_cache,    // [max_position, rot_dim]
    bool rope_is_neox,
    torch::Tensor& slot_mapping,          // [num_tokens] or [num_actual_tokens]
    torch::Tensor& kv_cache,              // [num_blocks, block_size,
                                          //  (kv_lora_rank + rot_dim)]
    const std::string& kv_cache_dtype,
    torch::Tensor& kv_cache_quant_scale) {
  const int64_t num_tokens = slot_mapping.size(0);
  const int64_t num_padded_tokens = q_pe.size(0);
  TORCH_CHECK(num_padded_tokens >= num_tokens);

  const int num_q_heads = q_pe.size(1);
  const int rot_dim = q_pe.size(2);
  const int kv_lora_rank = kv_c.size(1);

  TORCH_CHECK(positions.size(0) == num_padded_tokens);
  TORCH_CHECK(positions.dim() == 1);
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long);

  TORCH_CHECK(q_pe.dim() == 3);
  TORCH_CHECK(q_pe.size(0) == num_padded_tokens);
  TORCH_CHECK(q_pe.size(1) == num_q_heads);
  TORCH_CHECK(q_pe.size(2) == rot_dim);

  TORCH_CHECK(k_pe.dim() == 2);
  TORCH_CHECK(k_pe.size(0) == num_padded_tokens);
  TORCH_CHECK(k_pe.size(1) == rot_dim);
  TORCH_CHECK(k_pe.scalar_type() == q_pe.scalar_type());

  TORCH_CHECK(kv_c.dim() == 2);
  TORCH_CHECK(kv_c.size(0) == num_padded_tokens);
  TORCH_CHECK(kv_c.size(1) == kv_lora_rank);
  TORCH_CHECK(kv_c.scalar_type() == q_pe.scalar_type());

  TORCH_CHECK(rope_cos_sin_cache.size(1) == rot_dim);

  TORCH_CHECK(slot_mapping.size(0) == num_tokens);
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long);

  TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + rot_dim);
  TORCH_CHECK(kv_cache.dim() == 3);

  TORCH_CHECK(kv_cache_quant_scale.numel() == 1);
  TORCH_CHECK(kv_cache_quant_scale.scalar_type() == at::ScalarType::Float);

  int64_t q_pe_stride_token = q_pe.stride(0);
  int64_t q_pe_stride_head = q_pe.stride(1);
  int64_t k_pe_stride = k_pe.stride(0);
  int64_t kv_c_stride = kv_c.stride(0);

  int block_size = kv_cache.size(1);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  int rope_block_size = std::min(num_q_heads * rot_dim / 2, 512);
  int mla_block_size = kv_lora_rank;
  int thread_block_size =
      std::min(std::max(rope_block_size, mla_block_size), 512);

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(thread_block_size);

  const at::DeviceGuard device_guard(q_pe.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  DISPATCH_BY_KV_CACHE_DTYPE(q_pe.scalar_type(), kv_cache_dtype,
                             CALL_CONCAT_AND_CACHE_MLA_ROPE_FUSED);
}
