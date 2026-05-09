# SPDX-License-Identifier: Apache-2.0
"""Tests for XPU AWQ weight dequantization kernel."""

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401 — registers awq_dequantize op


def ref_awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                       zeros: torch.Tensor) -> torch.Tensor:
    """CPU reference implementation for awq_dequantize.

    AWQ uses interleaved nibble packing order [0, 4, 1, 5, 2, 6, 3, 7]:
    output position i uses nibble reverse_awq_order[i] from the packed int32.
    This matches CUDA dequantize_s4_to_fp16x2 and the Triton AWQ kernel.
    """
    in_c, qout_c = qweight.shape
    out_c = qout_c * 8
    G = in_c // scales.shape[0]
    output = torch.empty((in_c, out_c), dtype=scales.dtype)
    reverse_awq_order = [0, 4, 1, 5, 2, 6, 3, 7]

    for row in range(in_c):
        group = row // G
        for col in range(qout_c):
            w_packed = int(qweight[row, col].item()) & 0xFFFFFFFF
            z_packed = int(zeros[group, col].item()) & 0xFFFFFFFF
            for i in range(8):
                shift = reverse_awq_order[i] * 4
                w_nibble = (w_packed >> shift) & 0xF
                z_nibble = (z_packed >> shift) & 0xF
                scale_val = float(scales[group, col * 8 + i].item())
                output[row, col * 8 + i] = (w_nibble - z_nibble) * scale_val

    return output


@pytest.mark.parametrize(
    "in_c, qout_c, group_size",
    [
        (128, 16, 128),   # G == in_c (per-tensor group)
        (128, 16, 32),    # multiple groups
        (256, 32, 64),
        (512, 64, 128),
    ],
)
def test_awq_dequantize_accuracy(in_c: int, qout_c: int, group_size: int):
    """Verify XPU awq_dequantize matches reference output."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    torch.manual_seed(42)
    out_c = qout_c * 8
    num_groups = in_c // group_size

    qweight = torch.randint(0, 2**31 - 1, (in_c, qout_c), dtype=torch.int32)
    scales = torch.randn(num_groups, out_c, dtype=torch.float16)
    zeros = torch.randint(0, 2**31 - 1, (num_groups, qout_c), dtype=torch.int32)

    ref = ref_awq_dequantize(qweight, scales, zeros)

    qweight_xpu = qweight.to("xpu")
    scales_xpu = scales.to("xpu")
    zeros_xpu = zeros.to("xpu")

    out_xpu = torch.ops._C.awq_dequantize(qweight_xpu, scales_xpu, zeros_xpu,
                                           0, 0, 0)
    out_cpu = out_xpu.cpu().float()
    ref_f32 = ref.float()

    # fp16 arithmetic: allow small absolute tolerance
    torch.testing.assert_close(out_cpu, ref_f32, rtol=1e-3, atol=1e-3)


def test_awq_dequantize_shape():
    """Verify output shape is [in_c, out_c]."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    in_c, qout_c = 64, 8
    qweight = torch.zeros(in_c, qout_c, dtype=torch.int32, device="xpu")
    scales = torch.ones(1, qout_c * 8, dtype=torch.float16, device="xpu")
    zeros = torch.zeros(1, qout_c, dtype=torch.int32, device="xpu")

    out = torch.ops._C.awq_dequantize(qweight, scales, zeros, 0, 0, 0)
    assert out.shape == (in_c, qout_c * 8)
    assert out.dtype == torch.float16
    assert out.device.type == "xpu"
