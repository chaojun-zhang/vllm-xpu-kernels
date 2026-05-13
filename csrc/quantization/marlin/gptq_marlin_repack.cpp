/*
 * SYCL/XPU port of
 * vllm/csrc/quantization/marlin/gptq_marlin_repack.cu
 *
 * Repacks a GPTQ-packed weight matrix into Marlin format on Intel XPU.
 * Code structure mirrors the CUDA original: one kernel functor template
 * parameterised on <num_bits, has_perm, is_a_8bit>.
 */

#include <algorithm>
#include <cstdint>

#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include "ops.h"

namespace vllm {
namespace marlin {

// ---------------------------------------------------------------------------
// Constants (matching marlin.cuh)
// ---------------------------------------------------------------------------
static constexpr int tile_size = 16;
static constexpr int tile_k_size = tile_size;        // 16
static constexpr int tile_n_size = tile_k_size * 4;  // 64
static constexpr int repack_threads = 256;            // 8 warps × 32

static constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Kernel functor — mirrors gptq_marlin_repack_kernel<num_bits,has_perm,is_a_8bit>
//
// XPU simplification: CUDA uses a cp_async pipeline (repack_stages=8 stages).
// Since SYCL has no async global→local copy, we use synchronous loads + barriers
// (effectively pipeline depth 1).  Correctness is identical; throughput is
// fine because Intel GPU's memory subsystem hides latency differently.
// ---------------------------------------------------------------------------
template <int num_bits, bool has_perm, bool is_a_8bit>
class gptq_marlin_repack_kernel {
 public:
  // Derived tile constants (all constexpr → compile-time)
  static constexpr int pack_factor = 32 / num_bits;
  // is_a_8bit doubles the K-tile and halves the N-tile
  static constexpr int target_tile_n_size = tile_n_size / (is_a_8bit ? 2 : 1);
  static constexpr int target_tile_k_size = tile_k_size * (is_a_8bit ? 2 : 1);
  static constexpr int tile_ints = target_tile_k_size / pack_factor;
  // Perm region: perm_size int4 units (each int4 = 4 uint32s)
  static constexpr int perm_size = target_tile_k_size / 4;
  // Fetch tile dimensions
  static constexpr int stage_n_threads = target_tile_n_size / 4;
  static constexpr int stage_k_threads =
      has_perm ? target_tile_k_size : tile_ints;
  static constexpr int stage_size = stage_k_threads * stage_n_threads;
  // Local memory layout (uint32 units):  [perm | pipe]
  static constexpr int sh_perm_words = has_perm ? (perm_size * 4) : 0;
  static constexpr int sh_pipe_words = stage_size * 4;
  static constexpr int sh_total = sh_perm_words + sh_pipe_words;

 private:
  const uint32_t* b_q_weight_ptr_;
  const uint32_t* perm_ptr_;
  uint32_t* out_ptr_;
  int size_k_;
  int size_n_;
  sycl::local_accessor<uint32_t, 1> sh_;

 public:
  gptq_marlin_repack_kernel(const uint32_t* b_q_weight_ptr,
                             const uint32_t* perm_ptr,
                             uint32_t* out_ptr,
                             int size_k,
                             int size_n,
                             sycl::local_accessor<uint32_t, 1> sh)
      : b_q_weight_ptr_(b_q_weight_ptr),
        perm_ptr_(perm_ptr),
        out_ptr_(out_ptr),
        size_k_(size_k),
        size_n_(size_n),
        sh_(sh) {}

  void operator()(sycl::nd_item<1> item) const {
    int tid = static_cast<int>(item.get_local_id(0));
    int bid = static_cast<int>(item.get_group(0));
    int grid_x = static_cast<int>(item.get_group_range(0));

    int k_tiles = size_k_ / target_tile_k_size;
    int n_tiles = size_n_ / target_tile_n_size;
    int block_k_tiles = div_ceil(k_tiles, grid_x);
    int start_k_tile = bid * block_k_tiles;
    if (start_k_tile >= k_tiles) return;
    int finish_k_tile = sycl::min(start_k_tile + block_k_tiles, k_tiles);

    // Local memory pointers
    uint32_t* sh_base =
        sh_.template get_multi_ptr<sycl::access::decorated::no>().get();
    uint32_t* sh_perm_ptr = sh_base;
    uint32_t* sh_pipe_ptr = sh_base + sh_perm_words;

    for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
      // --- Load perm for this k-tile into local memory ---
      if constexpr (has_perm) {
        if (tid < perm_size) {
          // Each thread loads one int4 (4 uint32s) from perm
          int base = k_tile_id * target_tile_k_size + tid * 4;
          sh_perm_ptr[tid * 4 + 0] = perm_ptr_[base + 0];
          sh_perm_ptr[tid * 4 + 1] = perm_ptr_[base + 1];
          sh_perm_ptr[tid * 4 + 2] = perm_ptr_[base + 2];
          sh_perm_ptr[tid * 4 + 3] = perm_ptr_[base + 3];
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      for (int n_tile_id = 0; n_tile_id < n_tiles; n_tile_id++) {
        // --- Load weight tile into local memory ---
        int first_n = n_tile_id * target_tile_n_size;

        if (tid < stage_size) {
          int k_id = tid / stage_n_threads;
          int n_id = tid % stage_n_threads;
          int dst = (k_id * stage_n_threads + n_id) * 4;

          if constexpr (has_perm) {
            uint32_t src_k = sh_perm_ptr[k_id];
            int src_k_packed = static_cast<int>(src_k) / pack_factor;
            const uint32_t* src =
                b_q_weight_ptr_ + src_k_packed * size_n_ + first_n + n_id * 4;
            sh_pipe_ptr[dst + 0] = src[0];
            sh_pipe_ptr[dst + 1] = src[1];
            sh_pipe_ptr[dst + 2] = src[2];
            sh_pipe_ptr[dst + 3] = src[3];
          } else {
            int first_k_packed = k_tile_id * target_tile_k_size / pack_factor;
            const uint32_t* src = b_q_weight_ptr_ +
                                  (first_k_packed + k_id) * size_n_ +
                                  first_n + n_id * 4;
            sh_pipe_ptr[dst + 0] = src[0];
            sh_pipe_ptr[dst + 1] = src[1];
            sh_pipe_ptr[dst + 2] = src[2];
            sh_pipe_ptr[dst + 3] = src[3];
          }
        }

        item.barrier(sycl::access::fence_space::local_space);
        repack_tile(item, k_tile_id, n_tile_id, n_tiles, sh_perm_ptr,
                    sh_pipe_ptr);
        item.barrier(sycl::access::fence_space::local_space);
      }
    }
  }

 private:
  // Mirrors the repack_tile lambda in the CUDA kernel.
  void repack_tile(sycl::nd_item<1> item, int k_tile_id, int n_tile_id,
                   int n_tiles, uint32_t* sh_perm_ptr,
                   uint32_t* sh_pipe_ptr) const {
    int tid = static_cast<int>(item.get_local_id(0));
    int warp_id = tid / 32;
    int th_id = tid % 32;

    if (warp_id >= 4) return;  // Only 4 warps participate

    int tc_col = th_id / 4;  // 0..7
    int tc_row;
    if constexpr (is_a_8bit) {
      tc_row = (th_id % 4) * 4;  // 0, 4, 8, 12
    } else {
      tc_row = (th_id % 4) * 2;  // 0, 2, 4, 6
    }

    // cur_n: column in the local-memory tile (in uint32 units)
    int cur_n;
    if constexpr (is_a_8bit) {
      cur_n = (warp_id / 2) * 16 + tc_col;
    } else {
      cur_n = warp_id * 16 + tc_col;
    }

    // Row stride within the pipe local memory (in uint32 units)
    constexpr int sh_stride = target_tile_n_size;
    constexpr uint32_t mask = (1u << num_bits) - 1u;

    uint32_t vals[8];

    if constexpr (has_perm) {
      // has_perm implies !is_a_8bit
      constexpr int tc_offsets[4] = {0, 1, 8, 9};
      for (int i = 0; i < 4; i++) {
        int k_idx = tc_row + tc_offsets[i];
        uint32_t src_k = sh_perm_ptr[k_idx];
        uint32_t src_k_pos = src_k % static_cast<uint32_t>(pack_factor);

        uint32_t b1_val = sh_pipe_ptr[k_idx * sh_stride + cur_n];
        vals[i] = (b1_val >> (src_k_pos * num_bits)) & mask;

        uint32_t b2_val = sh_pipe_ptr[k_idx * sh_stride + cur_n + 8];
        vals[4 + i] = (b2_val >> (src_k_pos * num_bits)) & mask;
      }
    } else {
      uint32_t b1_vals[tile_ints];
      uint32_t b2_vals[tile_ints];

      for (int i = 0; i < tile_ints; i++) {
        if constexpr (is_a_8bit) {
          b1_vals[i] =
              sh_pipe_ptr[cur_n + sh_stride * i + (warp_id % 2) * 8];
        } else {
          b1_vals[i] = sh_pipe_ptr[cur_n + sh_stride * i];
          b2_vals[i] = sh_pipe_ptr[cur_n + 8 + sh_stride * i];
        }
      }

      constexpr int tc_offsets[4] = {0, 1, 8, 9};
      for (int i = 0; i < 4; i++) {
        int cur_elem;
        if constexpr (is_a_8bit) {
          cur_elem = tc_row + i;
        } else {
          cur_elem = tc_row + tc_offsets[i];
        }
        int cur_int = cur_elem / pack_factor;
        int cur_pos = cur_elem % pack_factor;

        vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        if constexpr (is_a_8bit) {
          vals[4 + i] =
              (b1_vals[cur_int + tile_ints / 2] >> (cur_pos * num_bits)) &
              mask;
        } else {
          vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        }
      }
    }

    // Write repacked values to output
    constexpr int tile_size_out =
        target_tile_k_size * target_tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size_out;

    if constexpr (!is_a_8bit && num_bits == 4) {
      // Non-a8bit 4-bit: interleave {0,2,4,6,1,3,5,7}
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
      uint32_t res = 0;
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }
      out_ptr_[out_offset + th_id * 4 + warp_id] = res;

    } else if constexpr (is_a_8bit && num_bits == 4) {
      // a8bit 4-bit: interleave {0,4,1,5,2,6,3,7}
      constexpr int pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      uint32_t res = 0;
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }
      out_ptr_[out_offset + th_id * 4 + warp_id] = res;

    } else {
      // 8-bit: two output words per thread, interleave {0,2,1,3} for !a8bit
      constexpr int pack_idx[4] = {0, 2, 1, 3};
      uint32_t res1 = 0;
      uint32_t res2 = 0;
      for (int i = 0; i < 4; i++) {
        const int ii = is_a_8bit ? i : pack_idx[i];
        res1 |= vals[ii] << (i * 8);
        res2 |= vals[4 + ii] << (i * 8);
      }
      out_ptr_[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr_[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  }
};

// ---------------------------------------------------------------------------
// Typed launch helper
// ---------------------------------------------------------------------------
template <int num_bits, bool has_perm, bool is_a_8bit>
static void launch_repack(const uint32_t* b_q_weight_ptr,
                           const uint32_t* perm_ptr,
                           uint32_t* out_ptr,
                           int size_k,
                           int size_n,
                           int blocks,
                           sycl::queue& q) {
  using Kernel = gptq_marlin_repack_kernel<num_bits, has_perm, is_a_8bit>;
  q.submit([&](sycl::handler& cgh) {
    auto sh = sycl::local_accessor<uint32_t, 1>(Kernel::sh_total, cgh);
    auto kfn = Kernel(b_q_weight_ptr, perm_ptr, out_ptr, size_k, size_n, sh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * repack_threads),
                          sycl::range<1>(repack_threads)),
        kfn);
  });
}

}  // namespace marlin
}  // namespace vllm

// ---------------------------------------------------------------------------
// Host function — mirrors CUDA gptq_marlin_repack()
// ---------------------------------------------------------------------------
torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight,
                                  torch::Tensor& perm,
                                  int64_t size_k,
                                  int64_t size_n,
                                  int64_t num_bits,
                                  bool is_a_8bit) {
  using namespace vllm::marlin;

  TORCH_CHECK(size_k % tile_k_size == 0, "size_k = ", size_k,
              " is not divisible by tile_k_size = ", tile_k_size);
  TORCH_CHECK(size_n % tile_n_size == 0, "size_n = ", size_n,
              " is not divisible by tile_n_size = ", tile_n_size);
  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8, got ", num_bits);

  int const pack_factor = 32 / static_cast<int>(num_bits);

  TORCH_CHECK(b_q_weight.size(0) == size_k / pack_factor,
              "b_q_weight.size(0) = ", b_q_weight.size(0),
              " != size_k / pack_factor = ", size_k / pack_factor);
  TORCH_CHECK(b_q_weight.size(1) == size_n,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " != size_n = ", size_n);

  TORCH_CHECK(b_q_weight.device().is_xpu(), "b_q_weight must be an XPU tensor");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight must be contiguous");
  TORCH_CHECK(b_q_weight.dtype() == at::kInt, "b_q_weight dtype must be int32");

  TORCH_CHECK(perm.device().is_xpu(), "perm must be an XPU tensor");
  TORCH_CHECK(perm.is_contiguous(), "perm must be contiguous");
  TORCH_CHECK(perm.dtype() == at::kInt, "perm dtype must be int32");

  // Device guard
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  // Allocate output (same shape formula as CUDA)
  auto options = torch::TensorOptions()
                     .dtype(b_q_weight.dtype())
                     .device(b_q_weight.device());
  torch::Tensor out = torch::empty(
      {size_k / tile_size, size_n * tile_size / pack_factor}, options);

  bool has_perm = perm.size(0) != 0;

  const uint32_t* b_q_weight_ptr =
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr());
  const uint32_t* perm_ptr =
      reinterpret_cast<const uint32_t*>(perm.data_ptr());
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr());

  sycl::queue& q = at::xpu::getCurrentXPUStream().queue();

  // Use EU count as block count (analogous to CUDA's SM count)
  int dev_idx = static_cast<int>(curDevice.index());
  auto* dev_prop = at::xpu::getDeviceProperties(dev_idx);
  int blocks = dev_prop ? std::max(1, static_cast<int>(dev_prop->gpu_eu_count))
                        : 32;

  // Dispatch on (num_bits, has_perm, is_a_8bit)
  if (num_bits == 4 && !has_perm && !is_a_8bit) {
    launch_repack<4, false, false>(b_q_weight_ptr, perm_ptr, out_ptr, size_k,
                                   size_n, blocks, q);
  } else if (num_bits == 4 && has_perm && !is_a_8bit) {
    launch_repack<4, true, false>(b_q_weight_ptr, perm_ptr, out_ptr, size_k,
                                  size_n, blocks, q);
  } else if (num_bits == 8 && !has_perm && !is_a_8bit) {
    launch_repack<8, false, false>(b_q_weight_ptr, perm_ptr, out_ptr, size_k,
                                   size_n, blocks, q);
  } else if (num_bits == 8 && has_perm && !is_a_8bit) {
    launch_repack<8, true, false>(b_q_weight_ptr, perm_ptr, out_ptr, size_k,
                                  size_n, blocks, q);
  } else if (num_bits == 4 && !has_perm && is_a_8bit) {
    launch_repack<4, false, true>(b_q_weight_ptr, perm_ptr, out_ptr, size_k,
                                  size_n, blocks, q);
  } else if (num_bits == 8 && !has_perm && is_a_8bit) {
    launch_repack<8, false, true>(b_q_weight_ptr, perm_ptr, out_ptr, size_k,
                                  size_n, blocks, q);
  } else {
    TORCH_CHECK(false, "Unsupported repack config: num_bits=", num_bits,
                ", has_perm=", has_perm, ", is_a_8bit=", is_a_8bit);
  }

  return out;
}
