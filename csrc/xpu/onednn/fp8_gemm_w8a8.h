#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

static inline void dnnl_matmul_w8a8_fp8(
    torch::Tensor& result,      // dst, [b, m, n]
    const torch::Tensor& mat1,  // src, [b, m, k]
    const torch::Tensor& mat2,  // quantized weight, [k, n] transpose
    bool is_nt,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m1_sc,
    const torch::Tensor& m2_sc) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  // Get scale dtypes early to determine quantization mode.
  auto m1_sc_dtype = m1_sc.scalar_type();
  auto m2_sc_dtype = m2_sc.scalar_type();

  // Debug: print actual C++ scalar types to verify XPU dtype behaviour.
  // Remove after root cause is confirmed.
  static std::atomic<int> _dbg_fp8_count{0};
  if (_dbg_fp8_count.fetch_add(1) < 4) {
    fprintf(stderr,
            "[DBG fp8_gemm] m1_sc scalar_type=%d (%s) m2_sc scalar_type=%d (%s)\n",
            static_cast<int>(m1_sc_dtype),
            c10::toString(m1_sc_dtype),
            static_cast<int>(m2_sc_dtype),
            c10::toString(m2_sc_dtype));
  }

  // On XPU, torch.empty(dtype=float8_e8m0fnu) may produce a Byte (uint8)
  // tensor at the C++ ATen level because the XPU backend does not always
  // register Float8_e8m0fnu as a first-class storage type.  The byte layout
  // is identical (1 byte per element), so treat Byte scale tensors as MXFP8
  // (Float8_e8m0fnu) throughout this function.
  auto is_mxfp8_scale_dtype = [](at::ScalarType dt) {
    return dt == at::ScalarType::Float8_e8m0fnu ||
           dt == at::ScalarType::Byte;
  };

  // block quant param (only for non-MXFP8 path).
  // For MXFP8 (Float8_e8m0fnu), weight scale uses OneDNN's expected [k//32, n]
  // layout, so m2_sc.size(1) == n != k//32 == m1_sc.size(1) for typical layers.
  // The is_block_quant size equality check must be skipped for MXFP8.
  bool is_block_quant = !is_mxfp8_scale_dtype(m1_sc_dtype) &&
                        (m1_sc.dim() == 2) && (m2_sc.dim() == 2) &&
                        (m1_sc.size(1) != 1) && (m2_sc.size(1) != 1);
  int64_t group_size = -1;
  if (is_block_quant) {
    TORCH_CHECK(
        m1_sc.size(1) == m2_sc.size(1),
        "Mismatch group size in input and weight.",
        m1_sc.size(1),
        " vs ",
        m2_sc.size(1));
    group_size = k / m1_sc.size(1);
  }

  // get joint dtypes
  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  auto wei_dtype = mat2.scalar_type();
  auto out_dtype = result.scalar_type();

  if (in_dtype == at::ScalarType::Float8_e5m2) {
    jd = out_dtype == at::ScalarType::BFloat16 ? joint_dtypes_t::f8_e5m2_bf16
                                               : joint_dtypes_t::f8_e5m2_f16;
  } else if (in_dtype == at::ScalarType::Float8_e4m3fn) {
    jd = out_dtype == at::ScalarType::BFloat16 ? joint_dtypes_t::f8_e4m3_bf16
                                               : joint_dtypes_t::f8_e4m3_f16;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for fp8 matmul: ", in_dtype);
  }

  // get bias type
  bias_type_t b_type = get_bias_type(bias, m, n);

  trans_type_t tt = trans_type_t::nn;
  if (is_nt) {
    // transpose mat2
    tt = trans_type_t::nt;
  }

  // get lda ldb and ldc
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false, "Unsupported input dimension for fp8 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (is_mxfp8_scale_dtype(m1_sc_dtype)) {
      TORCH_CHECK(
          is_mxfp8_scale_dtype(m2_sc_dtype),
          "Mismatched scale data types in mxfp8 matmul: ",
          m1_sc_dtype,
          " vs ",
          m2_sc_dtype);
      // Always use e8m0 for oneDNN — the tensor may appear as Byte on XPU
      // even when it was created as float8_e8m0fnu (identical byte layout).
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ (1 << 0) + (1 << 1),
          {1, 32},
          memory::data_type::e8m0);
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 0) + (1 << 1),
          {32, 1},
          memory::data_type::e8m0);
    } else {
      if (m1_sc.numel() == 1) {
        pattr.set_scales(
            DNNL_ARG_SRC,
            /* mask */ 0,
            {},
            get_onednn_dtype(m1_sc));
        /* per tensor quant */
      } else if (is_block_quant) {
        pattr.set_scales(
            DNNL_ARG_SRC,
            /* mask */ (1 << 0) + (1 << 1),
            {1, group_size},
            get_onednn_dtype(m1_sc));
        /* per block quant */
      } else {
        pattr.set_scales(
            DNNL_ARG_SRC,
            /* mask */ (1 << 0) + (1 << 1),
            {1, k},
            get_onednn_dtype(m1_sc));
        /* per token quant */
      }

      if (m2_sc.numel() == 1) {
        pattr.set_scales(
            DNNL_ARG_WEIGHTS,
            /* mask */ 0,
            {},
            get_onednn_dtype(m2_sc));
        /* per tensor quant */
      } else if (is_block_quant) {
        pattr.set_scales(
            DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 0) + (1 << 1),
            {group_size, group_size},
            get_onednn_dtype(m2_sc));
        /* per block quant */
      } else {
        pattr.set_scales(
            DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 1),
            {},
            get_onednn_dtype(m2_sc));
        /* per channel quant */
      }
    }
  };

  int arg_off = 0;

  // ************************************************************
  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int m1_sc_group_size = m1_sc.numel();
  int m2_sc_group_size = m2_sc.numel();
  int sc_group_size = (m1_sc_group_size << 8) | m2_sc_group_size;
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, dev_id, f_attr, sc_group_size);

  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      m2_sc.data_ptr(),
      [&]() {
        // Use e8m0 memory descriptor for MXFP8 scale (tensor may appear as
        // Byte on XPU even when created as float8_e8m0fnu).
        auto md = is_mxfp8_scale_dtype(m2_sc_dtype)
            ? memory::desc{get_onednn_dims(m2_sc), memory::data_type::e8m0,
                           get_onednn_strides(m2_sc)}
            : get_onednn_md(m2_sc);
        return make_onednn_memory(md, engine, m2_sc.data_ptr());
      });
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
        auto md = is_mxfp8_scale_dtype(m1_sc_dtype)
            ? memory::desc{get_onednn_dims(m1_sc), memory::data_type::e8m0,
                           get_onednn_strides(m1_sc)}
            : get_onednn_md(m1_sc);
        return make_onednn_memory(md, engine, m1_sc.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);

  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (get_shape(b_type) != bias_shape_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }
  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  auto qfp8_matmul_event =
      matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
}  // namespace oneDNN
