/*
 * Adapted from https://github.com/turboderp/exllamav2 and
 * https://github.com/qwopqwop200/GPTQ-for-LLaMa
 * SYCL/XPU port of vllm/csrc/quantization/gptq/q_gemm.cu (shuffle path only).
 */

#include <cstdint>

#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include "ops.h"
#include "quantization/gptq/qdq_2.h"
#include "quantization/gptq/qdq_3.h"
#include "quantization/gptq/qdq_4.h"
#include "quantization/gptq/qdq_8.h"

namespace vllm {
namespace gptq {

static constexpr int THREADS_X = 32;

// ---------------------------------------------------------------------------
// Shuffle kernel functors — reorder nibbles within each packed word.
// One work-item per output column n; loops over all k rows.
// Mirrors shuffle_4bit_kernel / shuffle_2bit_kernel / shuffle_3bit_kernel /
// shuffle_8bit_kernel from q_gemm.cu.
// ---------------------------------------------------------------------------

class shuffle_4bit_kernel {
 private:
  uint32_t* b_q_weight_;
  int size_k_;
  int size_n_;

 public:
  shuffle_4bit_kernel(uint32_t* b, int size_k, int size_n)
      : b_q_weight_(b), size_k_(size_k), size_n_(size_n) {}

  void operator()(sycl::nd_item<1> item) const {
    int n = static_cast<int>(item.get_global_id(0));
    if (n >= size_n_) return;
    uint32_t* b_ptr = b_q_weight_ + n;
    for (int k = 0; k < size_k_; k += 8) {
      shuffle_4bit_8(b_ptr, size_n_);
      b_ptr += size_n_;
    }
  }
};

class shuffle_8bit_kernel {
 private:
  uint32_t* b_q_weight_;
  int size_k_;
  int size_n_;

 public:
  shuffle_8bit_kernel(uint32_t* b, int size_k, int size_n)
      : b_q_weight_(b), size_k_(size_k), size_n_(size_n) {}

  void operator()(sycl::nd_item<1> item) const {
    int n = static_cast<int>(item.get_global_id(0));
    if (n >= size_n_) return;
    uint32_t* b_ptr = b_q_weight_ + n;
    for (int k = 0; k < size_k_; k += 4) {
      shuffle_8bit_4(b_ptr, size_n_);
      b_ptr += size_n_;
    }
  }
};

class shuffle_2bit_kernel {
 private:
  uint32_t* b_q_weight_;
  int size_k_;
  int size_n_;

 public:
  shuffle_2bit_kernel(uint32_t* b, int size_k, int size_n)
      : b_q_weight_(b), size_k_(size_k), size_n_(size_n) {}

  void operator()(sycl::nd_item<1> item) const {
    int n = static_cast<int>(item.get_global_id(0));
    if (n >= size_n_) return;
    uint32_t* b_ptr = b_q_weight_ + n;
    for (int k = 0; k < size_k_; k += 16) {
      shuffle_2bit_16(b_ptr, size_n_);
      b_ptr += size_n_;
    }
  }
};

class shuffle_3bit_kernel {
 private:
  uint32_t* b_q_weight_;
  int size_k_;
  int size_n_;

 public:
  shuffle_3bit_kernel(uint32_t* b, int size_k, int size_n)
      : b_q_weight_(b), size_k_(size_k), size_n_(size_n) {}

  void operator()(sycl::nd_item<1> item) const {
    int n = static_cast<int>(item.get_global_id(0));
    if (n >= size_n_) return;
    uint32_t* b_ptr = b_q_weight_ + n;
    for (int k = 0; k < size_k_; k += 32) {
      shuffle_3bit_32(b_ptr, size_n_);
      b_ptr += 3 * size_n_;
    }
  }
};

// ---------------------------------------------------------------------------
// make_sequential kernel functors — permute rows using q_perm.
// 2-D dispatch: dim-0 = output row, dim-1 = column within w2 (uint64 stride).
// Mirrors make_sequential_4bit_kernel / _2bit / _3bit / _8bit from q_gemm.cu.
// ---------------------------------------------------------------------------

class make_sequential_4bit_kernel {
 private:
  const uint32_t* w_;
  uint32_t* w_new_;
  const int* q_perm_;
  int w_width_;   // number of uint32_t columns in w/w_new
  int w2_stride_; // = w_width_ / 2 (uint64 columns)

 public:
  make_sequential_4bit_kernel(const uint32_t* w, uint32_t* w_new,
                              const int* q_perm, int w_width)
      : w_(w), w_new_(w_new), q_perm_(q_perm), w_width_(w_width),
        w2_stride_(w_width / 2) {}

  void operator()(sycl::nd_item<2> item) const {
    int w2_column = static_cast<int>(item.get_global_id(1));
    int w_new2_row = static_cast<int>(item.get_global_id(0));
    if (w2_column >= w2_stride_) return;

    const uint64_t* w2 = reinterpret_cast<const uint64_t*>(w_);
    uint64_t* w_new2 = reinterpret_cast<uint64_t*>(w_new_);

    int q_perm_idx = w_new2_row << 3;  // * 8
    uint64_t dst = 0;

#pragma unroll
    for (int i = 0; i < 8; i++) {
      int source_row = q_perm_[q_perm_idx++];
      int w2_row = source_row >> 3;
      int w2_subrow = source_row & 0x07;
      int w2_row_shift = w2_subrow << 2;
      int wnew2_row_shift = i << 2;

      uint64_t src = w2[w2_row * w2_stride_ + w2_column];
      src >>= w2_row_shift;
      src &= 0x0000000f0000000fULL;
      src <<= wnew2_row_shift;
      dst |= src;
    }
    w_new2[w_new2_row * w2_stride_ + w2_column] = dst;
  }
};

class make_sequential_2bit_kernel {
 private:
  const uint32_t* w_;
  uint32_t* w_new_;
  const int* q_perm_;
  int w_width_;
  int w2_stride_;

 public:
  make_sequential_2bit_kernel(const uint32_t* w, uint32_t* w_new,
                              const int* q_perm, int w_width)
      : w_(w), w_new_(w_new), q_perm_(q_perm), w_width_(w_width),
        w2_stride_(w_width / 2) {}

  void operator()(sycl::nd_item<2> item) const {
    int w2_column = static_cast<int>(item.get_global_id(1));
    int w_new2_row = static_cast<int>(item.get_global_id(0));
    if (w2_column >= w2_stride_) return;

    const uint64_t* w2 = reinterpret_cast<const uint64_t*>(w_);
    uint64_t* w_new2 = reinterpret_cast<uint64_t*>(w_new_);

    int q_perm_idx = w_new2_row << 4;  // * 16
    uint64_t dst = 0;

#pragma unroll
    for (int i = 0; i < 16; i++) {
      int source_row = q_perm_[q_perm_idx++];
      int w2_row = source_row >> 4;
      int w2_subrow = source_row & 0x0f;
      int w2_row_shift = w2_subrow << 1;
      int wnew2_row_shift = i << 1;

      uint64_t src = w2[w2_row * w2_stride_ + w2_column];
      src >>= w2_row_shift;
      src &= 0x0000000300000003ULL;
      src <<= wnew2_row_shift;
      dst |= src;
    }
    w_new2[w_new2_row * w2_stride_ + w2_column] = dst;
  }
};

class make_sequential_3bit_kernel {
 private:
  const uint32_t* w_;
  uint32_t* w_new_;
  const int* q_perm_;
  int w_width_;  // number of uint32_t columns

 public:
  make_sequential_3bit_kernel(const uint32_t* w, uint32_t* w_new,
                              const int* q_perm, int w_width)
      : w_(w), w_new_(w_new), q_perm_(q_perm), w_width_(w_width) {}

  void operator()(sycl::nd_item<2> item) const {
    int w_column = static_cast<int>(item.get_global_id(1));
    int block_row = static_cast<int>(item.get_global_id(0));
    if (w_column >= w_width_) return;

    int w_new_row = block_row * 3;
    int q_perm_idx = block_row << 5;  // * 32
    uint32_t dst[3] = {0, 0, 0};

#pragma unroll
    for (int i = 0; i < 32; i++) {
      int source_row = q_perm_[q_perm_idx++];
      int z_w = (source_row / 32) * 3;
      int z_mod = source_row % 32;
      int z_bit;

      if (z_mod != 10) {
        if (z_mod != 21) {
          z_bit = z_mod;
          if (z_bit > 21) {
            z_bit *= 3;
            z_bit -= 64;
            z_w += 2;
          } else if (z_bit > 10) {
            z_bit *= 3;
            z_bit -= 32;
            z_w += 1;
          } else {
            z_bit *= 3;
          }
        } else {
          z_w += 1;
        }
      }

      uint64_t src;
      if (z_mod == 10) {
        src = (w_[z_w * w_width_ + w_column] >> 30) |
              ((w_[(z_w + 1) * w_width_ + w_column] << 2) & 0x4);
      } else if (z_mod == 21) {
        src = (w_[z_w * w_width_ + w_column] >> 31) |
              ((w_[(z_w + 1) * w_width_ + w_column] << 1) & 0x6);
      } else {
        src = w_[z_w * w_width_ + w_column];
        src >>= z_bit;
        src &= 0x07;
      }

      z_w = 0;
      if (i != 10) {
        if (i != 21) {
          z_bit = i;
          if (z_bit > 21) {
            z_bit *= 3;
            z_bit -= 64;
            z_w += 2;
          } else if (z_bit > 10) {
            z_bit *= 3;
            z_bit -= 32;
            z_w += 1;
          } else {
            z_bit *= 3;
          }
        } else {
          z_w += 1;
        }
      }
      if (i == 10) {
        dst[z_w] |= (src & 0x03) << 30;
        dst[z_w + 1] |= static_cast<uint32_t>((src & 0x4) >> 2);
      } else if (i == 21) {
        dst[z_w] |= (src & 0x01) << 31;
        dst[z_w + 1] |= static_cast<uint32_t>((src & 0x6) >> 1);
      } else {
        dst[z_w] |= static_cast<uint32_t>(src << z_bit);
      }
    }
    w_new_[w_new_row * w_width_ + w_column] = dst[0];
    w_new_[(w_new_row + 1) * w_width_ + w_column] = dst[1];
    w_new_[(w_new_row + 2) * w_width_ + w_column] = dst[2];
  }
};

class make_sequential_8bit_kernel {
 private:
  const uint32_t* w_;
  uint32_t* w_new_;
  const int* q_perm_;
  int w_width_;
  int w2_stride_;

 public:
  make_sequential_8bit_kernel(const uint32_t* w, uint32_t* w_new,
                              const int* q_perm, int w_width)
      : w_(w), w_new_(w_new), q_perm_(q_perm), w_width_(w_width),
        w2_stride_(w_width / 2) {}

  void operator()(sycl::nd_item<2> item) const {
    int w2_column = static_cast<int>(item.get_global_id(1));
    int w_new2_row = static_cast<int>(item.get_global_id(0));
    if (w2_column >= w2_stride_) return;

    const uint64_t* w2 = reinterpret_cast<const uint64_t*>(w_);
    uint64_t* w_new2 = reinterpret_cast<uint64_t*>(w_new_);

    int q_perm_idx = w_new2_row << 2;  // * 4
    uint64_t dst = 0;

#pragma unroll
    for (int i = 0; i < 4; i++) {
      int source_row = q_perm_[q_perm_idx++];
      int w2_row = source_row >> 2;
      int w2_subrow = source_row & 0x03;
      int w2_row_shift = w2_subrow << 3;
      int wnew2_row_shift = i << 3;

      uint64_t src = w2[w2_row * w2_stride_ + w2_column];
      src >>= w2_row_shift;
      src &= 0x000000ff000000ffULL;
      src <<= wnew2_row_shift;
      dst |= src;
    }
    w_new2[w_new2_row * w2_stride_ + w2_column] = dst;
  }
};

// ---------------------------------------------------------------------------
// shuffle_exllama_weight — mirrors the CUDA function of the same name.
// ---------------------------------------------------------------------------

static void shuffle_exllama_weight(uint32_t* q_weight, int* q_perm, int height,
                                   int width, int bit, sycl::queue& q) {
  if (q_perm) {
    // Allocate temporary buffer on XPU.
    const int new_rows = height / 32 * bit;
    auto options = torch::TensorOptions()
                       .dtype(torch::kInt32)
                       .device(at::kXPU, at::xpu::current_device());
    torch::Tensor tmp_tensor =
        torch::empty({new_rows * width}, options);
    uint32_t* new_qweight =
        reinterpret_cast<uint32_t*>(tmp_tensor.data_ptr<int32_t>());

    const int x_blocks = (width / 2 + THREADS_X - 1) / THREADS_X;

    if (bit == 3) {
      // 3-bit: gridDim.y = height / 32
      const int y_blocks = height / 32;
      make_sequential_3bit_kernel kfn(q_weight, new_qweight, q_perm, width);
      q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(y_blocks, x_blocks * THREADS_X),
                sycl::range<2>(1, THREADS_X)),
            kfn);
      });
    } else {
      const int y_blocks = height / 32 * bit;
      if (bit == 2) {
        make_sequential_2bit_kernel kfn(q_weight, new_qweight, q_perm, width);
        q.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::nd_range<2>(
                  sycl::range<2>(y_blocks, x_blocks * THREADS_X),
                  sycl::range<2>(1, THREADS_X)),
              kfn);
        });
      } else if (bit == 4) {
        make_sequential_4bit_kernel kfn(q_weight, new_qweight, q_perm, width);
        q.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::nd_range<2>(
                  sycl::range<2>(y_blocks, x_blocks * THREADS_X),
                  sycl::range<2>(1, THREADS_X)),
              kfn);
        });
      } else {  // bit == 8
        make_sequential_8bit_kernel kfn(q_weight, new_qweight, q_perm, width);
        q.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::nd_range<2>(
                  sycl::range<2>(y_blocks, x_blocks * THREADS_X),
                  sycl::range<2>(1, THREADS_X)),
              kfn);
        });
      }
    }

    // Copy result back into q_weight and synchronize (mirrors cudaMemcpyAsync
    // + cudaDeviceSynchronize).
    const size_t bytes = static_cast<size_t>(new_rows) * width * sizeof(uint32_t);
    q.memcpy(q_weight, new_qweight, bytes);
    q.wait();
  }

  // Shuffle nibbles within each column.
  const int x_blocks = (width + THREADS_X - 1) / THREADS_X;
  const sycl::range<1> global_range(x_blocks * THREADS_X);
  const sycl::range<1> local_range(THREADS_X);

  if (bit == 2) {
    shuffle_2bit_kernel kfn(q_weight, height, width);
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    });
  } else if (bit == 3) {
    shuffle_3bit_kernel kfn(q_weight, height, width);
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    });
  } else if (bit == 8) {
    shuffle_8bit_kernel kfn(q_weight, height, width);
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    });
  } else {  // bit == 4
    shuffle_4bit_kernel kfn(q_weight, height, width);
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
    });
  }
}

}  // namespace gptq
}  // namespace vllm

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit) {
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto& q = at::xpu::getCurrentXPUStream().queue();

  int* perm_ptr = (q_perm.device().is_meta() || q_perm.numel() == 0)
                      ? nullptr
                      : q_perm.data_ptr<int>();

  vllm::gptq::shuffle_exllama_weight(
      reinterpret_cast<uint32_t*>(q_weight.data_ptr<int32_t>()),
      perm_ptr,
      static_cast<int>(q_weight.size(0)) * 32 / static_cast<int>(bit),
      static_cast<int>(q_weight.size(1)),
      static_cast<int>(bit),
      q);
}
