#!/usr/bin/env bash
# Build Unsloth's llama.cpp + the Inkling patches (macOS, Metal) and put the binaries in ./bin.
#
#   ./install.sh                  pinned Unsloth release the patches were made and verified on
#   ./install.sh <release-tag>    another Unsloth release, e.g. the one Studio runs now:
#                                 ./install.sh "$(jq -r .release_tag ~/.unsloth/llama.cpp/UNSLOTH_PREBUILT_INFO.json)"
#
# Unsloth Studio: Settings -> llama.cpp folder -> <this repo>/bin, speculative decoding Off, then reload the model.
set -euo pipefail

TAG=${1:-b12675-mix-d14fe45}
ROOT=$(cd "$(dirname "$0")" && pwd)
GIT=(git -c user.name=inkling -c user.email=inkling@localhost -c commit.gpgsign=false)
cd "$ROOT"

echo "==> Unsloth llama.cpp $TAG"
rm -rf llama.cpp && mkdir llama.cpp
curl -fsSL "https://github.com/unslothai/llama.cpp/releases/download/$TAG/llama.cpp-source-$TAG.tar.gz" |
  tar -xz -C llama.cpp --strip-components=1
cd llama.cpp
"${GIT[@]}" init -q && "${GIT[@]}" add -A && "${GIT[@]}" commit -qm "unsloth $TAG" && git tag base

echo "==> applying $(ls "$ROOT"/patches/*.patch | wc -l | tr -d ' ') patches"
if ! "${GIT[@]}" am -q -3 "$ROOT"/patches/*.patch; then
  cat >&2 <<EOF

A patch does not apply on $TAG; ./bin was not touched. To port the patches:
  cd llama.cpp            # fix the conflicts git lists above
  git add -A && git -c commit.gpgsign=false am --continue      # repeat until all are applied
  rm ../patches/*.patch && git format-patch --zero-commit --no-signature -o ../patches base
  cd .. && ./install.sh $TAG
EOF
  exit 1
fi

echo "==> build (static, Metal)"
FIX=()  # some Command Line Tools installs lack the libc++ headers: point at the SDK's copy
echo '#include <array>' | c++ -x c++ -fsyntax-only - 2>/dev/null ||
  FIX=(-DCMAKE_CXX_FLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1")
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON ${FIX[@]+"${FIX[@]}"} >/dev/null
cmake --build build -j "$(sysctl -n hw.ncpu)" --target llama-server llama-bench llama-perplexity test-backend-ops >/dev/null

echo "==> check: new Inkling ops, Metal vs CPU"
INK_EVAL=1 build/bin/test-backend-ops test -b MTL0 2>/dev/null | grep -E 'tests passed|FAIL'

# Rename into place (new inode) so a llama-server that Studio is running is not overwritten in place.
mkdir -p "$ROOT/bin"
for b in llama-server llama-bench llama-perplexity; do
  cp build/bin/$b "$ROOT/bin/.$b.new" && mv -f "$ROOT/bin/.$b.new" "$ROOT/bin/$b"
done
echo "$TAG" > "$ROOT/bin/VERSION"

echo
echo "Done: $ROOT/bin  (Unsloth $TAG + Inkling patches)"
echo "Unsloth Studio: Settings -> llama.cpp folder -> $ROOT/bin, speculative decoding Off, then reload the model."
