# SPDX-License-Identifier: Apache-2.0
"""Tests for XPU GPTQ weight shuffle kernel.

Mirrors vllm/tests/kernels/quantization/test_gptq.py.
"""

import ctypes

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401 — registers gptq_shuffle op


def _to_int32(v: int) -> int:
    """Reinterpret an unsigned uint32 value as a signed int32."""
    return ctypes.c_int32(v & 0xFFFFFFFF).value


# ---------------------------------------------------------------------------
# CPU reference implementations
# ---------------------------------------------------------------------------

def _shuffle_4bit_8_ref(val: int) -> int:
    """Reference: shuffle_4bit_8 — interleave low/high nibble halves."""
    qa = val & 0xFFFFFFFF
    qb = 0
    for i in range(4):
        qa0 = qa & 0x0F
        qa1 = (qa & 0xF0) >> 4
        qa = (qa >> 8) & 0xFFFFFFFF
        qb |= (qa1 << (i * 4 + 16))
        qb |= (qa0 << (i * 4))
    return qb & 0xFFFFFFFF


def _shuffle_2bit_16_ref(val: int) -> int:
    """Reference: shuffle_2bit_16 — interleave low/high 2-bit pairs."""
    qa = val & 0xFFFFFFFF
    qb = 0
    for i in range(8):
        qa0 = qa & 0x03
        qa1 = (qa & 0x0C) >> 2
        qa = (qa >> 4) & 0xFFFFFFFF
        qb |= (qa1 << (i * 2 + 16))
        qb |= (qa0 << (i * 2))
    return qb & 0xFFFFFFFF


def _shuffle_3bit_32_ref(qa: int, qb: int, qc: int) -> tuple:
    """Reference: shuffle_3bit_32 — repack three 32-bit words."""
    qa &= 0xFFFFFFFF
    qb &= 0xFFFFFFFF
    qc &= 0xFFFFFFFF

    qd = (qc >> 26) & 0xFFFFFFFF
    qc = ((qc << 4) | (qb >> 28)) & 0xFFFFFFFF
    qb = ((qb << 2) | (qa >> 30)) & 0xFFFFFFFF

    za = zb = zc = 0
    for i in range(5):
        t0 = qa & 0x07
        t1 = (qa & 0x38) >> 3
        qa = (qa >> 6) & 0xFFFFFFFF
        za |= (t0 << (i * 3))
        za |= (t1 << (i * 3 + 16))
    for i in range(5):
        t0 = qb & 0x07
        t1 = (qb & 0x38) >> 3
        qb = (qb >> 6) & 0xFFFFFFFF
        zb |= (t0 << (i * 3))
        zb |= (t1 << (i * 3 + 16))
    for i in range(5):
        t0 = qc & 0x07
        t1 = (qc & 0x38) >> 3
        qc = (qc >> 6) & 0xFFFFFFFF
        zc |= (t0 << (i * 3))
        zc |= (t1 << (i * 3 + 16))

    za |= ((qd & 0x01) >> 0) << 15
    zb |= ((qd & 0x02) >> 1) << 15
    zc |= ((qd & 0x04) >> 2) << 15
    za |= ((qd & 0x08) >> 3) << 31
    zb |= ((qd & 0x10) >> 4) << 31
    zc |= ((qd & 0x20) >> 5) << 31

    return (za & 0xFFFFFFFF, zb & 0xFFFFFFFF, zc & 0xFFFFFFFF)


def ref_gptq_shuffle_no_perm(q_weight: torch.Tensor, bit: int) -> torch.Tensor:
    """CPU reference for gptq_shuffle with empty perm (shuffle only)."""
    result = q_weight.clone()
    rows, cols = result.shape

    if bit == 4:
        # Each row of q_weight has `cols` uint32 words; each word covers 8 rows
        # of original weights. size_k = rows * 32 / bit = rows * 8
        size_k = rows * 32 // bit
        for n in range(cols):
            for k in range(0, size_k, 8):
                row = k // 8
                val = result[row, n].item() & 0xFFFFFFFF
                result[row, n] = _to_int32(_shuffle_4bit_8_ref(val))

    elif bit == 2:
        size_k = rows * 32 // bit
        for n in range(cols):
            for k in range(0, size_k, 16):
                row = k // 16
                val = result[row, n].item() & 0xFFFFFFFF
                result[row, n] = _to_int32(_shuffle_2bit_16_ref(val))

    elif bit == 8:
        # shuffle_8bit_4 is a no-op
        pass

    elif bit == 3:
        size_k = rows * 32 // bit
        for n in range(cols):
            for k in range(0, size_k, 32):
                row_base = (k // 32) * 3
                v0 = result[row_base,     n].item() & 0xFFFFFFFF
                v1 = result[row_base + 1, n].item() & 0xFFFFFFFF
                v2 = result[row_base + 2, n].item() & 0xFFFFFFFF
                r0, r1, r2 = _shuffle_3bit_32_ref(v0, v1, v2)
                result[row_base,     n] = _to_int32(r0)
                result[row_base + 1, n] = _to_int32(r1)
                result[row_base + 2, n] = _to_int32(r2)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight(rows: int, cols: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(-2_000_000, 2_000_000, (rows, cols), dtype=torch.int32)


def _rows_for_bit(height: int, bit: int) -> int:
    """Number of uint32 rows needed to store `height` original rows at `bit` bpw."""
    if bit == 3:
        # 32 values × 3 bits = 96 bits = 3 × uint32
        return (height // 32) * 3
    return (height * bit) // 32


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bit", [2, 3, 4, 8])
def test_gptq_shuffle_no_perm(bit: int):
    """gptq_shuffle with empty perm matches CPU reference (shuffle only)."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    height = 128   # original weight rows
    width  = 64    # output columns
    rows = _rows_for_bit(height, bit)

    weight_cpu = _make_weight(rows, width)
    weight_xpu = weight_cpu.to("xpu")
    perm = torch.empty((0,), device="xpu", dtype=torch.int32)

    torch.ops._C.gptq_shuffle(weight_xpu, perm, bit)

    expected = ref_gptq_shuffle_no_perm(weight_cpu, bit)
    result_cpu = weight_xpu.cpu()

    assert result_cpu.shape == weight_cpu.shape
    assert result_cpu.dtype == torch.int32
    torch.testing.assert_close(result_cpu, expected)


@pytest.mark.parametrize("bit", [2, 4, 8])
def test_gptq_shuffle_with_perm(bit: int):
    """gptq_shuffle with a valid q_perm changes the weight tensor."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    height = 64
    width  = 32
    rows = _rows_for_bit(height, bit)

    torch.manual_seed(0)
    weight_cpu = _make_weight(rows, width)
    weight_xpu = weight_cpu.to("xpu")

    perm_cpu = torch.randperm(height, dtype=torch.int32)
    perm_xpu = perm_cpu.to("xpu")

    # Keep a copy to verify the tensor is actually modified.
    original = weight_xpu.clone()
    torch.ops._C.gptq_shuffle(weight_xpu, perm_xpu, bit)

    # The result must differ from the original (permutation + shuffle).
    assert not torch.equal(weight_xpu.cpu(), original.cpu()), (
        "gptq_shuffle with non-trivial perm should change the weight tensor"
    )
    assert weight_xpu.shape == original.shape
    assert weight_xpu.dtype == torch.int32


@pytest.mark.parametrize("bit", [2, 3, 4, 8])
@pytest.mark.parametrize("height,width", [(64, 32), (128, 64), (256, 128)])
def test_gptq_shuffle_shape_dtype(bit: int, height: int, width: int):
    """gptq_shuffle preserves tensor shape and dtype."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    rows = _rows_for_bit(height, bit)
    weight = _make_weight(rows, width).to("xpu")
    perm = torch.empty((0,), device="xpu", dtype=torch.int32)

    torch.ops._C.gptq_shuffle(weight, perm, bit)

    assert weight.shape == (rows, width)
    assert weight.dtype == torch.int32
    assert weight.device.type == "xpu"


def test_gptq_shuffle_opcheck():
    """Mirrors test_gptq_shuffle_opcheck from vllm/tests/kernels/quantization/test_gptq.py."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    weight = torch.randint(
        -2_000_000, 2_000_000, (1792, 4096), device="xpu", dtype=torch.int32
    )
    perm = torch.empty((0,), device="xpu", dtype=torch.int32)
    bit = 4
    torch.ops._C.gptq_shuffle(weight, perm, bit)
    assert weight.shape == (1792, 4096)
    assert weight.dtype == torch.int32
