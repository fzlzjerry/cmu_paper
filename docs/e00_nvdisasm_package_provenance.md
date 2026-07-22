# E00 nvdisasm package provenance

- Date: 2026-07-22
- Scope: Phase 1 blocker B-002 remediation only
- Authorization: install exactly `cuda-nvdisasm-13-0=13.0.85-1`
- Experimental semantics changed: no
- Quality-only dependencies installed: none

## Pre-install verification

- Package manager: APT 2.8.3 with dpkg backend
- Native package architecture: `amd64`
- Configured source file: `/etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list`
- Repository: `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/`
- Repository signing key: `/usr/share/keyrings/cuda-archive-keyring.gpg`
- Candidate package: `cuda-nvdisasm-13-0=13.0.85-1`
- Package architecture: `amd64`
- Package filename: `cuda-nvdisasm-13-0_13.0.85-1_amd64.deb`
- Package size: 4,131,786 bytes
- Repository-metadata package SHA-256:
  `a0edae45d1debe83291254405afa1ee95a1f653e1fc59ea2cc6ffff6f49eefb5`
- Declared dependencies, pre-dependencies, and recommends: none
- Simulated transaction: 0 upgraded, 1 newly installed, 0 removed, and 16 not
  upgraded; the only install/configure action was the requested package
- Unrelated package changes: none. Two existing auto-removable packages were
  reported but were not removed.

## Installation

The only package-changing command was:

```text
apt-get install -y --no-install-recommends cuda-nvdisasm-13-0=13.0.85-1
```

APT downloaded 4,131,786 bytes from the configured NVIDIA repository and
installed only `cuda-nvdisasm-13-0:amd64` version `13.0.85-1`. The cached
package SHA-256 matched the repository metadata.

## Installed identity

- Installed package: `cuda-nvdisasm-13-0:amd64`
- Installed version: `13.0.85-1`
- E00 hermetic `command -v nvdisasm` path:
  `/usr/local/cuda-13.0/bin/nvdisasm`
- Ambient shell path before launcher sanitization: not present; the existing
  E00 launcher supplies `/usr/local/cuda-13.0/bin` explicitly
- Package ownership:
  `cuda-nvdisasm-13-0: /usr/local/cuda-13.0/bin/nvdisasm`
- Installed binary SHA-256:
  `3c27bded09bd877807207b62db8186a0a9a359d10311ab6e2c885f9b418c9f41`
- `dpkg -V cuda-nvdisasm-13-0`: no output, verification passed

Version output:

```text
nvdisasm: NVIDIA (R) CUDA disassembler
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Thu_Aug_14_07:20:41_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.85
Build cuda_13.0.r13.0/compiler.36400806_0
```

## Transaction boundary

No NVIDIA driver, CUDA runtime, compiler, PyTorch, Triton, profiler, benchmark,
model, adapter, CUDA kernel, measurement grid, quality dependency, or quality
runner was changed. The prior failed E00 run remains immutable; its manifest
SHA-256 after installation is still
`0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035`.
