# SPDX-License-Identifier: Apache-2.0
"""Tests for XPU gptq_marlin_repack kernel.

Mirrors vllm/tests/kernels/quantization/test_marlin_gemm.py
(test_gptq_marlin_repack section).

CPU reference implementations are ported from:
  vllm/vllm/model_executor/layers/quantization/utils/marlin_utils_test.py
  vllm/vllm/model_executor/layers/quantization/utils/quant_utils.py
"""

import numpy as np
import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401 — registers gptq_marlin_repack op

# Register FakeTensor implementation so opcheck's test_faketensor passes.
# Must happen after _C is imported (the op must be registered first).
if hasattr(torch.ops._C, "gptq_marlin_repack"):
    try:
        from torch.library import register_fake as _register_fake
    except ImportError:
        from torch.library import impl_abstract as _register_fake  # type: ignore

    @_register_fake("_C::gptq_marlin_repack")
    def _gptq_marlin_repack_fake(
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        size_k: torch.SymInt,
        size_n: torch.SymInt,
        num_bits: int,
        is_a_8bit: bool = False,
    ) -> torch.Tensor:
        pack_factor = 32 // num_bits
        marlin_tile_size = 16
        return torch.empty(
            (
                size_k // marlin_tile_size,
                size_n * marlin_tile_size // pack_factor,
            ),
            dtype=b_q_weight.dtype,
            device=b_q_weight.device,
        )


# ---------------------------------------------------------------------------
# Constants (from marlin.cuh / marlin_utils.py)
# ---------------------------------------------------------------------------
GPTQ_MARLIN_TILE = 16  # tile_size


# ---------------------------------------------------------------------------
# CPU reference: weight packing helpers (from quant_utils.py)
# ---------------------------------------------------------------------------

def _pack_rows(q_w: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack q_w (size_k, size_n) along rows into (size_k // pack_factor, size_n)."""
    size_k, size_n = q_w.shape
    pack_factor = 32 // num_bits
    q = q_w.cpu().numpy().astype(np.uint32)
    q_res = np.zeros((size_k // pack_factor, size_n), dtype=np.uint32)
    for i in range(pack_factor):
        q_res |= q[i::pack_factor, :] << (num_bits * i)
    return torch.from_numpy(q_res.view(np.int32))


# ---------------------------------------------------------------------------
# CPU reference: get_weight_perm (from marlin_utils_test.py)
# ---------------------------------------------------------------------------

def _get_weight_perm(num_bits: int, is_a_8bit: bool = False) -> torch.Tensor:
    perm_list: list[int] = []
    if is_a_8bit:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
                    4 * (i % 4 + 4),
                    4 * (i % 4 + 4) + 1,
                    4 * (i % 4 + 4) + 2,
                    4 * (i % 4 + 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(2):
                perm_list.extend([p + 512 * j for p in perm1])
    else:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)
    if num_bits == 4:
        interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7] if is_a_8bit
                               else [0, 2, 4, 6, 1, 3, 5, 7])
    else:  # num_bits == 8
        interleave = np.array([0, 1, 2, 3] if is_a_8bit else [0, 2, 1, 3])

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return torch.from_numpy(perm)


# ---------------------------------------------------------------------------
# CPU reference: marlin_weights (from marlin_utils_test.py)
# ---------------------------------------------------------------------------

def _marlin_permute_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    perm: torch.Tensor,
    tile: int = GPTQ_MARLIN_TILE,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    assert q_w.shape == (size_k, size_n)

    if is_a_8bit:
        q_w = q_w.reshape(
            (size_k // (tile * 2), tile * 2, size_n // tile, tile)
        )
    else:
        q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))

    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    return q_w


def _marlin_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    perm: torch.Tensor,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    q_w = _marlin_permute_weights(q_w, size_k, size_n, perm,
                                   is_a_8bit=is_a_8bit)
    pack_factor = 32 // num_bits
    q = q_w.cpu().numpy().astype(np.uint32)
    q_packed = np.zeros((q.shape[0], q.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q[:, i::pack_factor] << (num_bits * i)
    return torch.from_numpy(q_packed.view(np.int32))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_weights(size_k: int, size_n: int, num_bits: int) -> torch.Tensor:
    """Random integer weight matrix with values in [0, 2^num_bits)."""
    torch.manual_seed(42)
    return torch.randint(0, 2**num_bits, (size_k, size_n), dtype=torch.int32)


def _requires_xpu():
    has_xpu = torch.xpu.is_available()
    has_op = hasattr(torch.ops._C, "gptq_marlin_repack")
    return pytest.mark.skipif(
        not has_xpu or not has_op,
        reason="XPU device not available or gptq_marlin_repack op not built",
    )


# Sizes that satisfy marlin tile constraints:
# size_k must be divisible by tile_k_size = 16
# size_n must be divisible by tile_n_size = 64
SIZES = [
    (64, 64),
    (128, 64),
    (128, 128),
    (256, 128),
]

NUM_BITS = [4, 8]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size_k,size_n", SIZES)
@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("is_a_8bit", [False, True])
def test_gptq_marlin_repack_no_perm_cpu_ref(size_k, size_n, num_bits, is_a_8bit):
    """Verify CPU reference against itself (sanity check for shapes/dtypes)."""
    # a8bit requires size_k divisible by 32 and size_n divisible by 32
    if is_a_8bit and (size_k % 32 != 0 or size_n % 32 != 0):
        pytest.skip("is_a_8bit requires size_k%32==0 and size_n%32==0")

    q_w = _rand_weights(size_k, size_n, num_bits)
    weight_perm = _get_weight_perm(num_bits, is_a_8bit)
    marlin_ref = _marlin_weights(q_w, size_k, size_n, num_bits, weight_perm,
                                  is_a_8bit=is_a_8bit)

    pack_factor = 32 // num_bits
    expected_rows = size_k // GPTQ_MARLIN_TILE
    expected_cols = size_n * GPTQ_MARLIN_TILE // pack_factor
    assert marlin_ref.shape == (expected_rows, expected_cols), (
        f"marlin_ref shape {marlin_ref.shape} != ({expected_rows},{expected_cols})"
    )
    assert marlin_ref.dtype == torch.int32


@pytest.mark.parametrize("size_k,size_n", SIZES)
@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("is_a_8bit", [False, True])
@_requires_xpu()
def test_gptq_marlin_repack_no_perm(size_k, size_n, num_bits, is_a_8bit):
    """Kernel result (no perm) must equal CPU marlin_weights reference."""
    if is_a_8bit and (size_k % 32 != 0 or size_n % 32 != 0):
        pytest.skip("is_a_8bit requires size_k%32==0 and size_n%32==0")

    q_w = _rand_weights(size_k, size_n, num_bits)
    q_w_gptq = _pack_rows(q_w, num_bits).to("xpu")
    perm_empty = torch.empty(0, dtype=torch.int32, device="xpu")

    weight_perm = _get_weight_perm(num_bits, is_a_8bit)
    marlin_ref = _marlin_weights(q_w, size_k, size_n, num_bits, weight_perm,
                                  is_a_8bit=is_a_8bit).to("xpu")

    marlin_out = torch.ops._C.gptq_marlin_repack(
        q_w_gptq, perm_empty, size_k, size_n, num_bits, is_a_8bit
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(marlin_out, marlin_ref)


@pytest.mark.parametrize("size_k,size_n", SIZES)
@pytest.mark.parametrize("num_bits", NUM_BITS)
@_requires_xpu()
def test_gptq_marlin_repack_with_perm(size_k, size_n, num_bits):
    """Kernel result (with perm / has_perm=true) must equal CPU reference."""
    # has_perm is only for !is_a_8bit
    torch.manual_seed(0)
    q_w = _rand_weights(size_k, size_n, num_bits)

    # Random sort_indices (simulates act_order permutation)
    sort_indices = torch.randperm(size_k, dtype=torch.int32)

    # Pack the ORIGINAL (pre-sort) weights
    q_w_gptq = _pack_rows(q_w, num_bits).to("xpu")
    sort_indices_xpu = sort_indices.to("xpu")

    # CPU reference: apply sort then marlin_weights
    q_w_sorted = q_w[sort_indices, :]
    weight_perm = _get_weight_perm(num_bits, is_a_8bit=False)
    marlin_ref = _marlin_weights(q_w_sorted, size_k, size_n, num_bits,
                                  weight_perm, is_a_8bit=False).to("xpu")

    marlin_out = torch.ops._C.gptq_marlin_repack(
        q_w_gptq, sort_indices_xpu, size_k, size_n, num_bits, False
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(marlin_out, marlin_ref)


@pytest.mark.parametrize("size_k,size_n", SIZES)
@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("is_a_8bit", [False, True])
@_requires_xpu()
def test_gptq_marlin_repack_output_shape(size_k, size_n, num_bits, is_a_8bit):
    """Output tensor has the correct shape and dtype."""
    if is_a_8bit and (size_k % 32 != 0 or size_n % 32 != 0):
        pytest.skip("is_a_8bit requires size_k%32==0 and size_n%32==0")

    q_w = _rand_weights(size_k, size_n, num_bits)
    q_w_gptq = _pack_rows(q_w, num_bits).to("xpu")
    perm_empty = torch.empty(0, dtype=torch.int32, device="xpu")

    out = torch.ops._C.gptq_marlin_repack(
        q_w_gptq, perm_empty, size_k, size_n, num_bits, is_a_8bit
    )
    torch.xpu.synchronize()

    pack_factor = 32 // num_bits
    expected_shape = (size_k // GPTQ_MARLIN_TILE,
                      size_n * GPTQ_MARLIN_TILE // pack_factor)
    assert out.shape == torch.Size(expected_shape), (
        f"out.shape {out.shape} != {expected_shape}"
    )
    assert out.dtype == torch.int32
    assert out.device.type == "xpu"


@pytest.mark.parametrize("num_bits", NUM_BITS)
@pytest.mark.parametrize("is_a_8bit", [False, True])
@_requires_xpu()
def test_gptq_marlin_repack_opcheck(num_bits, is_a_8bit):
    """torch.library opcheck for schema / aliasing correctness."""
    from torch.testing._comparison import assert_close  # noqa: F401

    size_k, size_n = 64, 64
    if is_a_8bit and (size_k % 32 != 0 or size_n % 32 != 0):
        pytest.skip("is_a_8bit requires size_k%32==0 and size_n%32==0")

    q_w = _rand_weights(size_k, size_n, num_bits)
    q_w_gptq = _pack_rows(q_w, num_bits).to("xpu")
    perm_empty = torch.empty(0, dtype=torch.int32, device="xpu")

    torch.library.opcheck(
        torch.ops._C.gptq_marlin_repack,
        (q_w_gptq, perm_empty, size_k, size_n, num_bits, is_a_8bit),
        test_utils=["test_schema", "test_autograd_registration",
                    "test_faketensor"],
    )
