// Direct symmetric-memory DCP A2A + LSE reduce for XPU.
// Mirrors vLLM's CUDA op `_C.direct_dcp_a2a_lse_reduce`: each rank writes the
// partial output/LSE slices straight into peer buffers, publishes an epoch
// signal with release semantics, then waits on all peers and combines.
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>

#include "sycl_launch.h"
#include "utils.h"

namespace vllm {
// TEMPORARY DEBUG: remove before merging. Records kernel events into a
// host-pinned ring so a hung run can be inspected from the host.
#define DCP_A2A_DEBUG 1

namespace direct_dcp {

constexpr uint64_t kSpinLimit = 100000000;
constexpr float kLog2E = 1.4426950408889634f;

constexpr int64_t kTraceRecords = 1 << 16;
constexpr int64_t kTraceFields = 8;
// Event kinds: 1 IncrementEpoch(old,new) 2 SignalPeers(dst,epoch,parity)
// 3 WaitBegin(src,want,sig0,sig1) 4 WaitOk(src,want)
// 5 WaitSpin(src,want,cur) 6 WaitTimeout(src,want,sig0,sig1)

using trace_ref_t = sycl::atomic_ref<
    int64_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

inline void trace(
    int64_t* buf,
    int64_t kind,
    int64_t a = 0,
    int64_t b = 0,
    int64_t c = 0,
    int64_t d = 0) {
#ifdef DCP_A2A_DEBUG
  if (buf == nullptr) {
    return;
  }
  trace_ref_t counter(buf[0]);
  const int64_t idx = counter.fetch_add(1);
  if (idx >= kTraceRecords) {
    return;
  }
  int64_t* rec = buf + kTraceFields + idx * kTraceFields;
  rec[2] = a;
  rec[3] = b;
  rec[4] = c;
  rec[5] = d;
  rec[7] = idx;
  trace_ref_t(rec[0]).store(
      kind, sycl::memory_order::release, sycl::memory_scope::system);
#endif
}

int64_t* trace_buffer(sycl::queue& queue) {
#ifdef DCP_A2A_DEBUG
  static int64_t* buf = [&] {
    const size_t n = kTraceFields * (kTraceRecords + 1);
    auto* ptr = sycl::malloc_host<int64_t>(n, queue);
    std::memset(ptr, 0, n * sizeof(int64_t));
    return ptr;
  }();
  return buf;
#else
  return nullptr;
#endif
}

using signal_ref_t = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

template <typename T>
inline T* get_peer_ptr(const uint64_t* table, int64_t rank) {
  return reinterpret_cast<T*>(table[rank]);
}

inline void store_release_system(uint32_t* ptr, uint32_t value) {
  signal_ref_t ref(*ptr);
  ref.store(
      value, sycl::memory_order::release, sycl::memory_scope::system);
}

inline uint32_t load_acquire_system(const uint32_t* ptr) {
  signal_ref_t ref(*const_cast<uint32_t*>(ptr));
  return ref.load(sycl::memory_order::acquire, sycl::memory_scope::system);
}

inline bool wait_for_epoch(
    const uint32_t* ptr, uint32_t epoch, int64_t* trace_buf, int64_t src) {
  for (uint64_t spins = 0; spins < kSpinLimit; ++spins) {
    const uint32_t cur = load_acquire_system(ptr);
    if (cur == epoch) {
      return true;
    }
    if (trace_buf != nullptr && spins != 0 && (spins & 0x3fffff) == 0) {
      trace(trace_buf, 5, src, epoch, cur);
    }
  }
  return false;
}

inline int64_t find_sequence(
    const int32_t* query_start_loc, int64_t token_idx, int64_t num_seqs) {
  int64_t left = 0;
  int64_t right = num_seqs;
  while (left < right) {
    const int64_t mid = (left + right) / 2;
    if (query_start_loc[mid] <= token_idx) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

struct IncrementEpoch {
  int64_t* epoch;
  int64_t* trace_buf;
  void operator()(sycl::nd_item<1>) const {
    const int64_t old_epoch = epoch[0];
    epoch[0] += 1;
    trace(trace_buf, 1, old_epoch, epoch[0]);
  }
};

template <typename lse_t>
class DispatchOutputLse : public xpu::SyclKernelConfigConvention {
 public:
  DispatchOutputLse(
      const sycl::uint4* partial_output,
      const lse_t* partial_lse,
      const int32_t* seq_lens,
      const int32_t* query_start_loc,
      const uint64_t* peer_output_ptrs,
      const uint64_t* peer_lse_ptrs,
      const int64_t* epoch_ptr,
      int64_t world_size,
      int64_t rank,
      int64_t num_tokens,
      int64_t max_num_tokens,
      int64_t num_seqs,
      int64_t heads_per_rank,
      int64_t head_dim,
      int64_t output_token_stride,
      int64_t lse_token_stride,
      int64_t lse_head_stride)
      : partial_output_(partial_output),
        partial_lse_(partial_lse),
        seq_lens_(seq_lens),
        query_start_loc_(query_start_loc),
        peer_output_ptrs_(peer_output_ptrs),
        peer_lse_ptrs_(peer_lse_ptrs),
        epoch_ptr_(epoch_ptr),
        world_size_(world_size),
        rank_(rank),
        num_tokens_(num_tokens),
        max_num_tokens_(max_num_tokens),
        num_seqs_(num_seqs),
        heads_per_rank_(heads_per_rank),
        head_dim_(head_dim),
        output_token_stride_(output_token_stride),
        lse_token_stride_(lse_token_stride),
        lse_head_stride_(lse_head_stride) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    empty_kv_ = sycl::local_accessor<int32_t, 1>(1, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t tid = item.get_local_id(0);
    const int64_t block_size = item.get_local_range(0);
    const int64_t work_item = item.get_group(0);
    const int64_t destination_rank = work_item / num_tokens_;
    const int64_t token_idx = work_item - destination_rank * num_tokens_;

    if (tid == 0) {
      int32_t empty = 0;
      if (seq_lens_ != nullptr) {
        const int64_t seq_idx =
            find_sequence(query_start_loc_, token_idx, num_seqs_);
        empty = seq_lens_[seq_idx] == 0 ? 1 : 0;
      }
      empty_kv_[0] = empty;
    }
    sycl::group_barrier(item.get_group());
    const bool empty_kv = empty_kv_[0] != 0;

    const uint32_t epoch = static_cast<uint32_t>(epoch_ptr_[0]);
    const int64_t parity = static_cast<int64_t>(epoch & 1u);
    const int64_t destination_item =
        ((parity * world_size_ + rank_) * max_num_tokens_ + token_idx) *
        heads_per_rank_;
    const int64_t source_head = destination_rank * heads_per_rank_;

    if (!empty_kv) {
      auto* peer_output =
          get_peer_ptr<sycl::uint4>(peer_output_ptrs_, destination_rank);
      const int64_t vectors_per_item = heads_per_rank_ * head_dim_ / 8;
      const int64_t source_vector =
          (token_idx * output_token_stride_ + source_head * head_dim_) / 8;
      const int64_t destination_vector = destination_item * head_dim_ / 8;
      for (int64_t v = tid; v < vectors_per_item; v += block_size) {
        peer_output[destination_vector + v] = partial_output_[source_vector + v];
      }
    }

    float* peer_lse = get_peer_ptr<float>(peer_lse_ptrs_, destination_rank);
    const int64_t source_lse =
        token_idx * lse_token_stride_ + source_head * lse_head_stride_;
    for (int64_t h = tid; h < heads_per_rank_; h += block_size) {
      peer_lse[destination_item + h] =
          empty_kv ? -std::numeric_limits<float>::infinity()
                   : static_cast<float>(
                         partial_lse_[source_lse + h * lse_head_stride_]);
    }
  }

 private:
  const sycl::uint4* partial_output_;
  const lse_t* partial_lse_;
  const int32_t* seq_lens_;
  const int32_t* query_start_loc_;
  const uint64_t* peer_output_ptrs_;
  const uint64_t* peer_lse_ptrs_;
  const int64_t* epoch_ptr_;
  int64_t world_size_, rank_, num_tokens_, max_num_tokens_, num_seqs_;
  int64_t heads_per_rank_, head_dim_;
  int64_t output_token_stride_, lse_token_stride_, lse_head_stride_;
  sycl::local_accessor<int32_t, 1> empty_kv_;
};

struct SignalPeers {
  int64_t* trace_buf;
  const uint64_t* peer_signal_ptrs;
  const int64_t* epoch_ptr;
  int64_t world_size;
  int64_t rank;
  int64_t signal_stride;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t destination_rank = item.get_global_id(0);
    if (destination_rank >= world_size) {
      return;
    }
    const uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
    const int64_t parity = static_cast<int64_t>(epoch & 1u);
    auto* peer_signal = get_peer_ptr<uint32_t>(peer_signal_ptrs, destination_rank);
    store_release_system(
        peer_signal + (parity * world_size + rank) * signal_stride, epoch);
    trace(trace_buf, 2, destination_rank, epoch, parity);
  }
};

template <typename scalar_t>
class WaitLseCombine : public xpu::SyclKernelConfigConvention {
 public:
  WaitLseCombine(
      const scalar_t* received_output,
      const float* received_lse,
      const uint32_t* received_signal,
      const int64_t* epoch_ptr,
      scalar_t* combined_output,
      int64_t world_size,
      int64_t max_num_tokens,
      int64_t heads_per_rank,
      int64_t head_dim,
      bool is_lse_base_on_e,
      int64_t signal_stride,
      int64_t* trace_buf)
      : signal_stride_(signal_stride),
        trace_buf_(trace_buf),
        received_output_(received_output),
        received_lse_(received_lse),
        received_signal_(received_signal),
        epoch_ptr_(epoch_ptr),
        combined_output_(combined_output),
        world_size_(world_size),
        max_num_tokens_(max_num_tokens),
        heads_per_rank_(heads_per_rank),
        head_dim_(head_dim),
        is_lse_base_on_e_(is_lse_base_on_e) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    weights_ = sycl::local_accessor<float, 1>(world_size_, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t tid = item.get_local_id(0);
    const int64_t block_size = item.get_local_range(0);
    const int64_t work_item = item.get_group(0);
    const int64_t token_idx = work_item / heads_per_rank_;
    const int64_t head_idx = work_item - token_idx * heads_per_rank_;
    const uint32_t epoch = static_cast<uint32_t>(epoch_ptr_[0]);
    const int64_t parity = static_cast<int64_t>(epoch & 1u);
    constexpr float inf = std::numeric_limits<float>::infinity();

    if (tid == 0) {
      float lse_max = -inf;
      bool timed_out = false;
      for (int64_t src = 0; src < world_size_; ++src) {
        const int64_t source_item =
            ((parity * world_size_ + src) * max_num_tokens_ + token_idx) *
                heads_per_rank_ +
            head_idx;
        const bool traced = token_idx == 0 && head_idx == 0;
        int64_t* tb = traced ? trace_buf_ : nullptr;
        trace(
            tb,
            3,
            src,
            epoch,
            load_acquire_system(received_signal_ + src * signal_stride_),
            load_acquire_system(
                received_signal_ + (world_size_ + src) * signal_stride_));
        if (!wait_for_epoch(
                received_signal_ + (parity * world_size_ + src) * signal_stride_,
                epoch,
                tb,
                src)) {
          trace(
              tb,
              6,
              src,
              epoch,
              load_acquire_system(received_signal_ + src * signal_stride_),
              load_acquire_system(
                  received_signal_ + (world_size_ + src) * signal_stride_));
          sycl::ext::oneapi::experimental::printf(
              "direct DCP A2A timeout source=%ld token=%ld head=%ld epoch=%u\n",
              static_cast<long>(src),
              static_cast<long>(token_idx),
              static_cast<long>(head_idx),
              epoch);
          timed_out = true;
          break;
        }
        trace(tb, 4, src, epoch);
        float value = received_lse_[source_item];
        if (sycl::isnan(value) || value == inf) {
          value = -inf;
        }
        if (is_lse_base_on_e_) {
          value *= kLog2E;
        }
        weights_[src] = value;
        lse_max = sycl::fmax(lse_max, value);
      }
      if (timed_out) {
        // Surface the failure as NaNs instead of hanging the device.
        for (int64_t src = 0; src < world_size_; ++src) {
          weights_[src] = std::numeric_limits<float>::quiet_NaN();
        }
      } else {
        if (lse_max == -inf) {
          lse_max = 0.0f;
        }
        float lse_sum = 0.0f;
        for (int64_t src = 0; src < world_size_; ++src) {
          lse_sum += sycl::exp2(weights_[src] - lse_max);
        }
        const float inverse_lse_sum = lse_sum > 0.0f ? 1.0f / lse_sum : 0.0f;
        for (int64_t src = 0; src < world_size_; ++src) {
          weights_[src] = sycl::exp2(weights_[src] - lse_max) * inverse_lse_sum;
        }
      }
    }
    sycl::group_barrier(item.get_group());

    for (int64_t d = tid; d < head_dim_; d += block_size) {
      float accumulator = 0.0f;
      for (int64_t src = 0; src < world_size_; ++src) {
        const float weight = weights_[src];
        if (weight == 0.0f) {
          continue;
        }
        const int64_t source_item =
            ((parity * world_size_ + src) * max_num_tokens_ + token_idx) *
                heads_per_rank_ +
            head_idx;
        accumulator +=
            static_cast<float>(received_output_[source_item * head_dim_ + d]) *
            weight;
      }
      combined_output_[work_item * head_dim_ + d] =
          static_cast<scalar_t>(accumulator);
    }
  }

 private:
  int64_t signal_stride_;
  int64_t* trace_buf_;
  const scalar_t* received_output_;
  const float* received_lse_;
  const uint32_t* received_signal_;
  const int64_t* epoch_ptr_;
  scalar_t* combined_output_;
  int64_t world_size_, max_num_tokens_, heads_per_rank_, head_dim_;
  bool is_lse_base_on_e_;
  sycl::local_accessor<float, 1> weights_;
};

inline const uint64_t* ptr_table(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(
      (t.scalar_type() == at::kLong || t.scalar_type() == at::kUInt64) &&
          t.is_contiguous() && t.device().is_xpu(),
      name,
      " must be a contiguous int64/uint64 XPU tensor");
  return reinterpret_cast<const uint64_t*>(t.data_ptr());
}

}  // namespace direct_dcp
}  // namespace vllm

void direct_dcp_a2a_lse_reduce(
    torch::Tensor const& partial_output,
    torch::Tensor const& partial_lse,
    std::optional<torch::Tensor> const& seq_lens,
    std::optional<torch::Tensor> const& query_start_loc,
    torch::Tensor const& peer_output_ptrs,
    torch::Tensor const& peer_lse_ptrs,
    torch::Tensor const& peer_signal_ptrs,
    torch::Tensor& received_output,
    torch::Tensor& received_lse,
    torch::Tensor& received_signal,
    torch::Tensor& epoch,
    torch::Tensor& combined_output,
    int64_t world_size,
    int64_t rank,
    int64_t max_num_tokens,
    bool is_lse_base_on_e) {
  using namespace vllm::direct_dcp;
  const at::DeviceGuard device_guard(partial_output.device());
  TORCH_CHECK(
      partial_output.is_xpu() && partial_lse.is_xpu(),
      "partial output and LSE must be XPU tensors");
  const auto output_dtype = partial_output.scalar_type();
  TORCH_CHECK(
      output_dtype == at::kHalf || output_dtype == at::kBFloat16,
      "direct DCP A2A only supports FP16 and BF16 output");
  const auto lse_dtype = partial_lse.scalar_type();
  TORCH_CHECK(
      lse_dtype == at::kFloat || lse_dtype == at::kHalf ||
          lse_dtype == at::kBFloat16,
      "partial LSE must be FP32, FP16, or BF16");
  TORCH_CHECK(
      partial_output.dim() == 3 && partial_lse.dim() == 2,
      "expected output [T,H,D] and LSE [T,H]");
  TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  const int64_t num_tokens = partial_output.size(0);
  const int64_t total_heads = partial_output.size(1);
  const int64_t head_dim = partial_output.size(2);
  const int64_t output_token_stride = partial_output.stride(0);
  const int64_t lse_token_stride = partial_lse.stride(0);
  const int64_t lse_head_stride = partial_lse.stride(1);
  TORCH_CHECK(
      partial_output.stride(2) == 1 && partial_output.stride(1) == head_dim &&
          output_token_stride >= total_heads * head_dim &&
          output_token_stride % 8 == 0,
      "partial output must have packed heads and an aligned token stride");
  TORCH_CHECK(
      num_tokens > 0 && num_tokens <= max_num_tokens,
      "token count exceeds symmetric buffer capacity");
  TORCH_CHECK(
      total_heads % world_size == 0,
      "attention heads must divide evenly across DCP ranks");
  TORCH_CHECK(
      partial_lse.size(0) == num_tokens && partial_lse.size(1) == total_heads,
      "LSE shape must match attention output");
  TORCH_CHECK(
      head_dim % 8 == 0, "head_dim must be divisible by 8 for 16-byte stores");
  const int64_t heads_per_rank = total_heads / world_size;
  TORCH_CHECK(
      combined_output.scalar_type() == output_dtype &&
          combined_output.is_contiguous() && combined_output.is_xpu(),
      "combined output must match the contiguous XPU input");
  TORCH_CHECK(
      combined_output.size(0) == num_tokens &&
          combined_output.size(1) == heads_per_rank &&
          combined_output.size(2) == head_dim,
      "combined output has the wrong shape");
  TORCH_CHECK(
      received_output.scalar_type() == output_dtype &&
          received_lse.scalar_type() == at::kFloat &&
          received_signal.scalar_type() == at::kInt,
      "invalid symmetric staging dtypes");
  // Signal slots may be padded (one cache line each) so that writers on
  // different devices never share a line; the stride is inferred from size.
  const int64_t signal_slots = 2 * world_size;
  TORCH_CHECK(
      received_signal.numel() >= signal_slots &&
          received_signal.numel() % signal_slots == 0,
      "received_signal must hold a multiple of 2 * world_size elements");
  const int64_t signal_stride = received_signal.numel() / signal_slots;
  TORCH_CHECK(
      epoch.scalar_type() == at::kLong && epoch.numel() >= 1,
      "epoch must be an int64 tensor");

  const int32_t* seq_lens_ptr = nullptr;
  const int32_t* query_start_loc_ptr = nullptr;
  int64_t num_seqs = 0;
  TORCH_CHECK(
      seq_lens.has_value() == query_start_loc.has_value(),
      "seq_lens and query_start_loc must be provided together");
  if (seq_lens.has_value()) {
    TORCH_CHECK(
        seq_lens->is_xpu() && seq_lens->dim() == 1 &&
            seq_lens->stride(0) == 1 && seq_lens->scalar_type() == at::kInt,
        "seq_lens must be a contiguous 1-D int32 XPU tensor");
    TORCH_CHECK(
        query_start_loc->is_xpu() && query_start_loc->dim() == 1 &&
            query_start_loc->stride(0) == 1 &&
            query_start_loc->scalar_type() == at::kInt,
        "query_start_loc must be a contiguous 1-D int32 XPU tensor");
    num_seqs = seq_lens->size(0);
    TORCH_CHECK(
        num_seqs > 0 && query_start_loc->size(0) == num_seqs + 1,
        "query_start_loc must contain one boundary per sequence");
    seq_lens_ptr = seq_lens->const_data_ptr<int32_t>();
    query_start_loc_ptr = query_start_loc->const_data_ptr<int32_t>();
  }

  const uint64_t* output_ptrs = ptr_table(peer_output_ptrs, "peer_output_ptrs");
  const uint64_t* lse_ptrs = ptr_table(peer_lse_ptrs, "peer_lse_ptrs");
  const uint64_t* signal_ptrs = ptr_table(peer_signal_ptrs, "peer_signal_ptrs");

  auto& queue = vllm::xpu::vllmGetQueue();
  constexpr int64_t kExchangeThreads = 256;
  constexpr int64_t kCombineThreads = 128;
  int64_t* epoch_ptr = epoch.data_ptr<int64_t>();

  int64_t* trace_buf = trace_buffer(queue);
  vllm::xpu::sycl_kernel_submit(queue, 1, 1, IncrementEpoch{epoch_ptr, trace_buf});

  auto launch_dispatch = [&](auto lse_tag) {
    using lse_t = decltype(lse_tag);
    vllm::xpu::sycl_kernel_submit(
        queue,
        world_size * num_tokens * kExchangeThreads,
        kExchangeThreads,
        DispatchOutputLse<lse_t>(
            reinterpret_cast<const sycl::uint4*>(partial_output.data_ptr()),
            reinterpret_cast<const lse_t*>(partial_lse.data_ptr()),
            seq_lens_ptr,
            query_start_loc_ptr,
            output_ptrs,
            lse_ptrs,
            epoch_ptr,
            world_size,
            rank,
            num_tokens,
            max_num_tokens,
            num_seqs,
            heads_per_rank,
            head_dim,
            output_token_stride,
            lse_token_stride,
            lse_head_stride));
  };
  if (lse_dtype == at::kFloat) {
    launch_dispatch(float{});
  } else if (lse_dtype == at::kBFloat16) {
    launch_dispatch(sycl::ext::oneapi::bfloat16{});
  } else {
    launch_dispatch(sycl::half{});
  }

  const int64_t signal_groups =
      (world_size + kExchangeThreads - 1) / kExchangeThreads;
  vllm::xpu::sycl_kernel_submit(
      queue,
      signal_groups * kExchangeThreads,
      kExchangeThreads,
      SignalPeers{
          trace_buf, signal_ptrs, epoch_ptr, world_size, rank, signal_stride});

  auto launch_combine = [&](auto scalar_tag) {
    using scalar_t = decltype(scalar_tag);
    vllm::xpu::sycl_kernel_submit(
        queue,
        num_tokens * heads_per_rank * kCombineThreads,
        kCombineThreads,
        WaitLseCombine<scalar_t>(
            reinterpret_cast<const scalar_t*>(received_output.data_ptr()),
            received_lse.const_data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(
                received_signal.const_data_ptr<int32_t>()),
            epoch_ptr,
            reinterpret_cast<scalar_t*>(combined_output.data_ptr()),
            world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
            is_lse_base_on_e,
            signal_stride,
            trace_buf));
  };
  if (output_dtype == at::kBFloat16) {
    launch_combine(sycl::ext::oneapi::bfloat16{});
  } else {
    launch_combine(sycl::half{});
  }
}

// TEMPORARY DEBUG: alias of the host-pinned trace ring (int64 CPU tensor).
// Element 0 is the record counter; records are 8 int64 starting at index 8.
at::Tensor direct_dcp_a2a_trace(bool reset) {
  auto& queue = vllm::xpu::vllmGetQueue();
  int64_t* buf = vllm::direct_dcp::trace_buffer(queue);
  TORCH_CHECK(buf != nullptr, "DCP_A2A_DEBUG is not enabled");
  const int64_t n = vllm::direct_dcp::kTraceFields *
                    (vllm::direct_dcp::kTraceRecords + 1);
  if (reset) {
    std::memset(buf, 0, n * sizeof(int64_t));
  }
  return torch::from_blob(buf, {n}, torch::kInt64);
}
