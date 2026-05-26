# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import tests.register_ops as ops
from tests.utils import seed_everything

DTYPES = [torch.half, torch.bfloat16]
FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
QUANT_DTYPES = [torch.float8_e4m3fn, torch.int8]
NUM_TOKENS = [1, 7, 83, 512]
HIDDEN_SIZES = [16, 128, 512, 4096]
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# --- per-block-quant params ---
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [64, *VEC_HIDDEN_SIZES, 2048, 5120]],
    *[(16, i) for i in [64, *VEC_HIDDEN_SIZES, 5120]],
    *[(128, i) for i in [64, *VEC_HIDDEN_SIZES]],
    *[(512, i) for i in [64, 5120]],
]
GROUP_SIZES = [64, 128]
IS_SCALE_TRANSPOSED = [False, True]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "HIDDEN_SIZES": [16],
    },
    "test_silu_and_mul_per_block_quant": {
        "num_tokens, hidden_size": [(1, 128)],
        "group_size": [128],
        "dtype": [torch.float16],
        "quant_dtype": [torch.float8_e4m3fn],
        "device": [XPU_DEVICES[0]],
    },
    "test_silu_and_mul_per_block_quant_shapes": {
        "device": [XPU_DEVICES[0]],
    },
    "test_silu_and_mul_per_block_quant_edge_cases": {
        "hidden_size": [1024],
        "batch_size": [1],
        "device": [XPU_DEVICES[0]],
    },
}


def ref_silu_and_mul_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference implementation: SiLU+Mul then static FP8 quant."""
    d = x.shape[-1] // 2
    silu_mul_out = F.silu(x[..., :d]) * x[..., d:]

    fp8_max = torch.finfo(fp8_dtype).max
    inv_scale = 1.0 / scale.item()
    result = (silu_mul_out.float() * inv_scale).clamp(-fp8_max, fp8_max)
    return result.to(fp8_dtype)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)

    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    scale = torch.tensor([0.5], dtype=torch.float32, device=device)

    # Reference
    ref_out = ref_silu_and_mul_quant(x, scale, fp8_dtype)

    # Fused kernel
    d = x.shape[-1] // 2
    out = torch.empty(num_tokens, d, dtype=fp8_dtype, device=device)
    ops.silu_and_mul_quant(out, x, scale)

    assert out.dtype == fp8_dtype
    assert out.shape == ref_out.shape
    torch.testing.assert_close(
        out.to(dtype=torch.float32),
        ref_out.to(dtype=torch.float32),
    )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_quant_vs_separate(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)

    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    scale = torch.tensor([0.5], dtype=torch.float32, device=device)

    # Separate ops
    d = x.shape[-1] // 2
    silu_mul_out = torch.empty(num_tokens, d, dtype=dtype, device=device)
    ops.silu_and_mul(silu_mul_out, x)

    separate_out = torch.empty(num_tokens, d, dtype=fp8_dtype, device=device)
    ops.static_scaled_fp8_quant(separate_out, silu_mul_out, scale)

    # Fused kernel
    fused_out = torch.empty(num_tokens, d, dtype=fp8_dtype, device=device)
    ops.silu_and_mul_quant(fused_out, x, scale)

    assert fused_out.dtype == fp8_dtype
    assert fused_out.shape == separate_out.shape
    torch.testing.assert_close(
        fused_out.to(dtype=torch.float32),
        separate_out.to(dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# silu_and_mul_per_block_quant tests
# ---------------------------------------------------------------------------

def _per_token_group_quant(
    x: torch.Tensor,
    group_size: int,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group quantization reference (pure PyTorch, fp8 or int8)."""
    x_f32 = x.float()
    num_groups = x.shape[-1] // group_size
    grouped = x_f32.view(*x.shape[:-1], num_groups, group_size)
    group_max = grouped.abs().amax(dim=-1)

    if quant_dtype == torch.float8_e4m3fn:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        scale = group_max.clamp(min=torch.finfo(torch.float32).tiny) / fp8_max
        x_q = (grouped / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    else:  # int8
        scale = group_max.clamp(min=1e-10) / 127.0
        x_q = (grouped / scale.unsqueeze(-1)).round().clamp(-128, 127)

    return x_q.view_as(x).to(quant_dtype), scale.to(torch.float32)


def ref_silu_and_mul_per_block_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    silu_out = (F.silu(gate) * up).contiguous()
    return _per_token_group_quant(silu_out, group_size, quant_dtype)


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("is_scale_transposed", IS_SCALE_TRANSPOSED)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    is_scale_transposed: bool,
    quant_dtype: torch.dtype,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    if hidden_size % group_size != 0:
        return

    torch.random.manual_seed(seed)
    torch.set_default_device(device)
    scale = 1 / hidden_size
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype) * scale

    ref_out, ref_scales = ref_silu_and_mul_per_block_quant(
        x, quant_dtype, group_size
    )
    ops_out, ops_scales = ops.silu_and_mul_per_block_quant(
        x, group_size, quant_dtype, None, is_scale_transposed
    )
    ops_scales_row = ops_scales.contiguous()

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    assert ops_scales.dtype == torch.float32
    assert not torch.isnan(ops_out.float()).any(), "output contains NaN"
    assert not torch.isinf(ops_out.float()).any(), "output contains Inf"
    assert not torch.isnan(ops_scales_row).any(), "scales contain NaN"
    assert not torch.isinf(ops_scales_row).any(), "scales contain Inf"

    torch.testing.assert_close(ref_scales, ops_scales_row, rtol=1e-5, atol=1e-5)

    ref_scales_expanded = ref_scales.repeat_interleave(group_size, dim=1)
    ops_scales_expanded = ops_scales_row.repeat_interleave(group_size, dim=1)
    ref_deq = ref_out.float() * ref_scales_expanded
    ops_deq = ops_out.float() * ops_scales_expanded
    torch.testing.assert_close(ref_deq, ops_deq, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant_shapes(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    group_size: int,
    device: str,
) -> None:
    torch.set_default_device(device)
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    num_groups = hidden_size // group_size

    out, scales = ops.silu_and_mul_per_block_quant(
        x, group_size, torch.float8_e4m3fn, is_scale_transposed=False
    )
    assert out.shape == (num_tokens, hidden_size)
    assert scales.shape == (num_tokens, num_groups)

    out, scales_t = ops.silu_and_mul_per_block_quant(
        x, group_size, torch.float8_e4m3fn, is_scale_transposed=True
    )
    assert out.shape == (num_tokens, hidden_size)
    assert scales_t.shape == (num_tokens, num_groups)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("batch_size", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [1024, 5120, 14336])
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant_edge_cases(
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
    device: str,
) -> None:
    torch.set_default_device(device)
    x = torch.randn(batch_size, hidden_size * 2, dtype=dtype)

    out, scales = ops.silu_and_mul_per_block_quant(
        x, 128, torch.float8_e4m3fn, is_scale_transposed=False
    )

    assert out.shape == (batch_size, hidden_size)
    assert out.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert not torch.isnan(out.float()).any()
    assert not torch.isnan(scales).any()
    assert not torch.isinf(scales).any()
