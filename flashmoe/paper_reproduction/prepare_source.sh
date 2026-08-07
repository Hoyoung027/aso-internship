#!/bin/bash

set -euo pipefail

PAPER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$PAPER_DIR/.." && pwd)
FLASHMOE_SOURCE=${FLASHMOE_SOURCE:-/home/hybyun0207/FlashMoE}
PATCH_FILE=$PAPER_DIR/patches/sm120_fp32.patch
GENERATED_ROOT=$PAPER_DIR/generated

if [[ ! -d "$FLASHMOE_SOURCE/.git" ]]; then
  echo "FlashMoE git source not found: $FLASHMOE_SOURCE" >&2
  exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch not found: $PATCH_FILE" >&2
  exit 1
fi
for executable in git tar patch sha256sum flock; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable not found: $executable" >&2
    exit 1
  fi
done

commit=$(git -C "$FLASHMOE_SOURCE" rev-parse HEAD)
patch_hash=$(sha256sum "$PATCH_FILE" | awk '{print $1}')
source_key=${commit:0:12}-${patch_hash:0:12}
target=$GENERATED_ROOT/source-$source_key
marker=$target/.paper_source_manifest

mkdir -p "$GENERATED_ROOT"
exec 9>"$GENERATED_ROOT/.prepare.lock"
flock 9

if [[ -f "$marker" ]] \
  && grep -qx "flashmoe_commit=$commit" "$marker" \
  && grep -qx "patch_sha256=$patch_hash" "$marker"; then
  printf '%s\n' "$target"
  exit 0
fi

tmp=$(mktemp -d "$GENERATED_ROOT/.prepare.XXXXXX")
cleanup() {
  if [[ -n "${tmp:-}" && -d "$tmp" ]]; then
    rm -rf -- "$tmp"
  fi
}
trap cleanup EXIT

git -C "$FLASHMOE_SOURCE" archive "$commit" | tar -x -C "$tmp"
patch --directory="$tmp" --strip=1 --forward --input="$PATCH_FILE" >&2

mkdir -p "$tmp/csrc/cmake"
if [[ ! -d "$FLASHMOE_SOURCE/csrc/cmake/cache" ]]; then
  echo "Upstream CPM cache is missing: $FLASHMOE_SOURCE/csrc/cmake/cache" >&2
  exit 1
fi
ln -s "$FLASHMOE_SOURCE/csrc/cmake/cache" "$tmp/csrc/cmake/cache"

{
  echo "flashmoe_commit=$commit"
  echo "patch_sha256=$patch_hash"
  echo "generated_at=$(date --iso-8601=seconds)"
} > "$tmp/.paper_source_manifest"

if [[ -e "$target" ]]; then
  echo "Generated source exists without a matching manifest: $target" >&2
  exit 1
fi
mv "$tmp" "$target"
tmp=

printf '%s\n' "$target"
