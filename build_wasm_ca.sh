#!/bin/bash
# Build the Neural CA WASM demo. Embeds the trained rule (nca.bin).
set -e; cd "$(dirname "$0")"
EMSDK="${EMSDK:-/c/prog/emsdk/emsdk}"
export EM_CONFIG="$EMSDK/.emscripten"
export PATH="$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$EMSDK/node/22.16.0_64bit/bin:$EMSDK/python/3.13.3_64bit:$PATH"
NET="${NET:-nca.bin}"
OUT="${OUT:-wasmdist}"; mkdir -p "$OUT"
emcc -O3 -std=c++17 wasm_ca.cpp \
  --embed-file "$NET@nca.bin" \
  -sMODULARIZE=1 -sEXPORT_NAME=createCA -sENVIRONMENT=web -sALLOW_MEMORY_GROWTH=1 -sFORCE_FILESYSTEM=1 \
  -sEXPORTED_FUNCTIONS=_ca_init,_ca_n,_ca_reset,_ca_step,_ca_damage,_ca_rgba,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=cwrap,HEAPU8 \
  -o "$OUT/ca.js"
echo "built $OUT/ca.js (+.wasm) with NET=$NET"
