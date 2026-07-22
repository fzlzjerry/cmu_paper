#!/usr/bin/bash
set -euo pipefail

startup_keys=(
    BASH_ENV
    ENV
    LD_LIBRARY_PATH
    LD_PRELOAD
    PYTHONHOME
    PYTHONPATH
)
for key in "${startup_keys[@]}"; do
    if [[ -n "${!key-}" ]]; then
        echo "E00 refused before Python startup: unsafe environment key is set: ${key}" >&2
        exit 2
    fi
done

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" != */* ]]; then
    echo "E00 refused before Python startup: launcher path is not explicit" >&2
    exit 2
fi
script_dir="$(cd -- "${script_path%/*}" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
e00_cuda_home="/usr/local/cuda-13.0"
project_python="${repo_root}/.venv/bin/python"
expected_python="/usr/bin/python3.12"
expected_python_sha256="1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"

if [[ ! -x "${project_python}" ]]; then
    echo "E00 refused before Python startup: project interpreter is missing" >&2
    exit 2
fi
observed_python="$(/usr/bin/readlink -f -- "${project_python}")"
if [[ "${observed_python}" != "${expected_python}" ]]; then
    echo "E00 refused before Python startup: project interpreter target mismatch" >&2
    exit 2
fi
checksum_line="$(/usr/bin/sha256sum -- "${observed_python}")"
observed_python_sha256="${checksum_line%% *}"
if [[ "${observed_python_sha256}" != "${expected_python_sha256}" ]]; then
    echo "E00 refused before Python startup: project interpreter hash mismatch" >&2
    exit 2
fi

cd -- "${repo_root}"
exec /usr/bin/env -i \
    LC_ALL=C \
    LANG=C \
    TZ=UTC \
    PATH="${e00_cuda_home}/bin:/usr/bin:/bin" \
    CUDA_HOME="${e00_cuda_home}" \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/c++ \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONOPTIMIZE=0 \
    PYTHONHASHSEED=0 \
    "${project_python}" "${repo_root}/preflight/run_preflight.py"
