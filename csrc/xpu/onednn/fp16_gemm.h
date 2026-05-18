#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

// Plain fp16 GEMM: C[m,n] = A[m,k] x B[k,n]  (NN layout, no transpose)
static inline void dnnl_matmul_fp16(
    torch::Tensor& result,      // dst,    [m, n]
    const torch::Tensor& mat1,  // src,    [m, k]
    const torch::Tensor& mat2,  // weight, [k, n]
    const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat1.dim() == 2, "dnnl_matmul_fp16: mat1 must be 2D");
  TORCH_CHECK(mat2.dim() == 2, "dnnl_matmul_fp16: mat2 must be 2D");
  TORCH_CHECK(
      mat1.scalar_type() == at::ScalarType::Half &&
          mat2.scalar_type() == at::ScalarType::Half,
      "dnnl_matmul_fp16: both inputs must be float16");

  const int m = mat1.size(0);
  const int k = mat1.size(1);
  const int n = mat2.size(1);

  const int64_t lda = mat1.stride(0);  // k for contiguous [m, k]
  const int64_t ldb = mat2.stride(0);  // n for contiguous [k, n]
  const int64_t ldc = result.stride(0);

  bias_type_t b_type = get_bias_type(bias, m, n);

  auto f_attr = [&](primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
  };

  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  auto engine = GpuEngineManager::Instance().get_engine(
      at::Device(at::kXPU, dev_id));

  auto& matmul_ext =
      matmul_primitive_create_and_cache<joint_dtypes_t::f16, decltype(f_attr)>(
          trans_type_t::nn,
          b_type,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          dev_id,
          f_attr,
          /*scale_group_size=*/1,
          /*zp_group_size=*/1);

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(4);
  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (get_shape(b_type) != bias_shape_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }

  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad =
      at::empty({scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  matmul_ext.execute(strm, engine, std::move(arg_handles), /*slot_off=*/0);
}

}  // namespace oneDNN
