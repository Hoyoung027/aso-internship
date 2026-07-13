#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /path/to/flashinfer /path/to/output-root" >&2
  exit 2
fi

repo=$1
output_root=$2

if [[ ! -d "$repo/.git" ]]; then
  echo "not a FlashInfer git checkout: $repo" >&2
  exit 1
fi

mkdir -p "$output_root"

# ea569640은 TVM-FFI 전환 커밋의 바로 이전 parent로 torch.ops loader를 쓴다.
if [[ ! -e "$output_root/pybind" ]]; then
  git -C "$repo" worktree add "$output_root/pybind" ea569640
fi

# 86d3e136은 동일 시점의 TVM-FFI 전환 직후 커밋이다.
if [[ ! -e "$output_root/tvm-ffi" ]]; then
  git -C "$repo" worktree add "$output_root/tvm-ffi" 86d3e136
fi

echo "Created worktrees:"
echo "  PyTorch binding: $output_root/pybind"
echo "  TVM-FFI:         $output_root/tvm-ffi"
echo
echo "Initialize submodules and create a separate virtualenv in each worktree."
