## About

`vllm-xpu-kernels` is a [vLLM](https://github.com/vllm-project/vllm) plugin that provides optimized custom kernels for Intel discrete GPUs (known as **XPU** in PyTorch). It leverages Intel's [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) / SYCL programming model and integrates tightly with [oneDNN](https://github.com/oneapi-src/oneDNN) and [CUTLASS-SYCL](https://github.com/intel/cutlass-sycl) to deliver high-performance LLM inference operators on Intel GPU hardware.

**Target hardware:** Intel PVC (Ponte Vecchio) and BMG-G21 discrete GPU architectures.

**Stack:** PyTorch 2.8 (XPU build) · Intel oneAPI 2025.1 · SYCL/DPC++

---

## Project Architecture

```
vllm-xpu-kernels/
├── csrc/                        # C++/SYCL kernel source code
│   ├── activation.cpp           # SiLU, GELU, and fused activation kernels
│   ├── cache.cpp                # KV-cache reshape and copy operations
│   ├── layernorm.cpp            # RMS Norm and fused Add+RMSNorm
│   ├── pos_encoding_kernels.cpp # Rotary position embeddings (NeoX & Llama style)
│   ├── torch_bindings.cpp       # PyTorch custom-op registration for _C extension
│   ├── quantization/fp8/        # FP8 static/dynamic quantization operators
│   ├── flash_attn/              # Flash Attention v2 (CUTLASS-SYCL backend)
│   ├── moe/                     # Mixture-of-Experts: TopK, softmax, alignment
│   └── xpu/                     # XPU-exclusive kernels
│       ├── cutlass_kernels/     # CUTLASS-based grouped GEMM for MoE experts
│       ├── lora/                # LoRA shrink / expand operations
│       ├── onednn/              # oneDNN FP8 GEMM and INT4 GEMM
│       └── sycl/                # Pure SYCL kernels (e.g. DeepSeek scaling RoPE)
├── vllm_xpu_kernels/            # Python package (wrappers + auto-registration)
│   ├── __init__.py
│   ├── flash_attn_interface.py  # Flash Attention v2 Python API
│   ├── fused_moe_interface.py   # Fused MoE Python API
│   └── quantization/           # Quantization helper utilities
├── tests/                       # pytest test suite
├── benchmark/                   # Performance benchmark scripts
├── cmake/                       # CMake toolchain and utility modules
├── tools/                       # Dev-tooling (linting, type-checking, etc.)
├── CMakeLists.txt               # Top-level build configuration
├── setup.py                     # PEP 517 build backend (CMake-based)
├── pyproject.toml               # Build system metadata
├── requirements.txt             # Python development dependencies
└── Dockerfile.xpu               # Docker image for CI / reproducible builds
```

### Compiled Extensions

The build system produces **four** shared library extensions:

| Extension | Sources | Purpose |
|-----------|---------|---------|
| `_C` | `csrc/*.cpp`, `csrc/quantization/` | Core ops: cache, layernorm, activation, FP8 quant, rotary embeddings |
| `_vllm_fa2_C` | `csrc/flash_attn/` | Flash Attention v2 (CUTLASS-SYCL, BMG-G21 optimized) |
| `_moe_C` | `csrc/moe/` | MoE routing: grouped TopK, softmax, alignment/sum reduction |
| `_xpu_C` | `csrc/xpu/` | LoRA ops, CUTLASS grouped GEMM, oneDNN FP8/INT4 GEMM, DeepSeek RoPE |

All extensions target the `spir64` / `spir64_gen` SYCL backend and are built with Python Stable ABI (`abi3`).

---

## Kernel Implementations

### Core Operators (`_C`)

| Kernel | API | Notes |
|--------|-----|-------|
| RMS Norm | `rms_norm`, `fused_add_rms_norm` | Used by transformer blocks |
| KV-cache | `reshape_and_cache` | Supports various cache dtypes |
| Activation | `silu_and_mul`, `gelu_and_mul`, `gelu_new_and_mul`, `gelu_fast_and_mul` | Fused activation + multiply |
| Rotary Embedding | `rotary_embedding`, `batched_rotary_embedding` | NeoX and Llama-2 variants |
| FP8 Quantization | `static_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`, `dynamic_per_token_scaled_fp8_quant` | Per-tensor and per-token |

### Flash Attention v2 (`_vllm_fa2_C`)

Variable-length (prefill + decode) attention kernel built on [CUTLASS-SYCL](https://github.com/intel/cutlass-sycl). Optimized for BMG-G21 with 256 GRF-per-thread. Exposed via `vllm_xpu_kernels.flash_attn_interface`.

### Mixture-of-Experts (`_moe_C`)

| Kernel | Notes |
|--------|-------|
| `grouped_topk` | Expert selection per token |
| `fused_grouped_topk` | Fused version for higher throughput |
| `topk_softmax` | Softmax over expert logits |
| `moe_align_block_size` | Aligns token counts to GEMM block boundaries |
| `moe_sum` | Reduces expert outputs into final hidden state |

### XPU-Exclusive Ops (`_xpu_C`)

| Kernel | Backend | Notes |
|--------|---------|-------|
| Grouped GEMM | CUTLASS-SYCL | Expert-wise batched GEMM for MoE |
| FP8 GEMM | oneDNN (static link) | INT8 accumulation, FP8 weight/activation |
| INT4 GEMM | oneDNN (static link) | 4-bit weight dequantize + GEMM |
| LoRA shrink | SYCL | Low-rank adapter weight multiplication |
| LoRA expand | SYCL | Low-rank adapter output projection |
| DeepSeek scaling RoPE | SYCL | Yarn-scaled rotary embeddings for DeepSeek models |

---

## Getting Started

### Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.9 – 3.12 |
| PyTorch (XPU build) | 2.8.0+xpu |
| Intel oneAPI DL Essentials | 2025.1 |
| CMake | ≥ 3.26 |
| Ninja | any recent |

### Prepare

1. Install [Intel oneAPI 2025.1 Deep Learning Essentials](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

2. Source the oneAPI environment:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

3. Create a virtual environment and install Python dependencies:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

### Build & Install

**Development (editable) install:**
```bash
pip install --extra-index-url=https://download.pytorch.org/whl/xpu -e . -v
# faster, skipping build isolation:
pip install --no-build-isolation -e . -v
```

**System-wide install:**
```bash
pip install --extra-index-url=https://download.pytorch.org/whl/xpu .
pip install --no-build-isolation .
```

**Build wheel only (output in `dist/`):**
```bash
pip wheel --extra-index-url=https://download.pytorch.org/whl/xpu .
pip wheel --no-build-isolation .
```

**Incremental rebuild after code changes:**
```bash
python3 -m build --wheel --no-isolation
```

**Direct CMake build:**
```bash
mkdir build && cd build
cmake -G Ninja \
      -DVLLM_PYTHON_EXECUTABLE=$(which python3) \
      -DCMAKE_INSTALL_PREFIX=.. \
      -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain.cmake \
      ..
cmake --build . --target install
```

Useful environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug` / `RelWithDebInfo` / `Release` |
| `MAX_JOBS` | CPU count | Parallel compilation jobs |
| `FETCHCONTENT_BASE_DIR` | `<build>/fetchcontent` | Where CMake downloads CUTLASS-SYCL |
| `VERBOSE` | unset | Set to `1` for verbose build output |

### How It Works

```
pip install -e .
  └─► setup.py (cmake_build_ext)
        └─► CMake → icpx/SYCL compilation
              └─► _C.abi3.so, _vllm_fa2_C.abi3.so,
                  _moe_C.abi3.so, _xpu_C.abi3.so
                    └─► placed in vllm_xpu_kernels/

import vllm_xpu_kernels._C   ← triggers PyTorch custom-op registration
```

All four `.so` files self-register their PyTorch custom operators on import, so vLLM can call them transparently.

### How to Use in vLLM

Please refer to the temporary branch https://github.com/jikunshang/vllm/tree/xpu_kernel to install and test vLLM, which replaces the `rms_norm` kernel from IPEX with `vllm-xpu-kernels`.

---

## Running Tests

```bash
# Install test dependencies (included in requirements.txt)
pytest -v tests/
```

Individual test modules:

```bash
pytest tests/test_activation.py
pytest tests/test_layernorm.py
pytest tests/test_cache.py
pytest tests/test_rotary_embedding.py
pytest tests/test_fp8_quant.py
pytest tests/test_lora_ops.py
pytest tests/flash_attn/test_flash_attn_varlen_func.py
pytest tests/fused_moe/test_fused_moe.py
```

Tests compare XPU kernel outputs against CPU reference implementations and validate numerical accuracy within configurable tolerances.

---

## Benchmarks

Performance benchmark scripts are located in `benchmark/`. Run them after building and installing the package:

```bash
python benchmark/benchmark_flash_attn.py
python benchmark/benchmark_moe.py
```

---

## Development

### Code Style

Pre-commit hooks are configured in `.pre-commit-config.yaml`:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Included checks: code formatting, SPDX license header validation, import format enforcement, shell script linting, and MyPy type checking.

### Type Checking

```bash
bash tools/mypy.sh
```

### Docker

A reproducible build environment is provided via `Dockerfile.xpu`:

```bash
docker build -f Dockerfile.xpu -t vllm-xpu-kernels .
docker run --device /dev/dri -it vllm-xpu-kernels bash
```

---

## Why Static Linking for oneDNN (DNNL)?

We chose to **statically link oneDNN (DNNL)** rather than using it as a shared library for the following reasons:

#### 1. Version Compatibility

Static linking ensures our application always uses the exact version of DNNL. With shared libraries, there is a risk that system-installed versions might be incompatible or introduce subtle bugs due to API/ABI changes.

#### 2. Performance Consistency

By linking statically, we avoid potential performance variability introduced by different builds or configurations of DNNL that might be present on the host system.

#### 3. Avoiding Runtime Errors

Using shared libraries requires correct paths and environment setup (`LD_LIBRARY_PATH` on Linux). Static linking avoids issues where DNNL cannot be found or loaded at runtime.

#### 4. Aligning with PyTorch

One key reason to use static linking is to maintain consistency with the PyTorch ecosystem. PyTorch itself statically links libraries like DNNL to ensure deterministic and reliable behavior across different environments.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
