#!/usr/bin/env bash
set -euo pipefail

N="${1:-20000000}"
RUNS="${RUNS:-3}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
BIN_DIR="$ROOT/.bench/bin"
GEN_DIR="$ROOT/.bench/gen"
mkdir -p "$BIN_DIR" "$GEN_DIR"

ARKNET_BIN="${ARKNET_BIN:-../bin/arknet-0.1.0-alpha-linux-amd64}"

C_SRC="$ROOT/bench_modadd.c"
PY_SRC="$ROOT/bench_modadd.py"
ARK_SRC_TEMPLATE="$ROOT/bench_modadd.ark"
ARK_SRC_GEN="$GEN_DIR/bench_modadd_n_${N}.ark"

time_cmd() {
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -f "real=%E user=%U sys=%S maxrss=%MKB" "$@"
  else
    time "$@"
  fi
}

require_file() {
  local p="$1"
  [[ -f "$p" ]] || { echo "[ERROR] Missing file: $p" >&2; exit 1; }
}

require_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $c" >&2; exit 1; }
}

extract_checksum() {
  local s="$1"
  local v
  v="$(printf '%s\n' "$s" | sed -nE 's/.*([0-9]+)[[:space:]]*$/\1/p' | tail -n1)"
  [[ -n "$v" ]] || return 1
  printf '%s\n' "$v"
}

build_ark_source_for_n() {
  local in="$1"
  local out="$2"
  local n="$3"

  python3 - "$in" "$out" "$n" <<'PY'
import re
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
n_value = sys.argv[3]

src = src_path.read_text(encoding="utf-8")

# Replace only the first `let n = <int>;` occurrence.
new_src, count = re.subn(r'\blet\s+n\s*=\s*\d+\s*;', f'let n = {n_value};', src, count=1)

if count != 1:
    print(f"[ERROR] Could not find a single 'let n = <int>;' in {src_path}", file=sys.stderr)
    sys.exit(1)

out_path.write_text(new_src, encoding="utf-8")
PY
}

run_quiet() {
  "$@" 2>/dev/null
}

echo "== Building =="

require_cmd gcc
require_cmd python3
require_file "$C_SRC"
require_file "$PY_SRC"
require_file "$ARK_SRC_TEMPLATE"

gcc -O3 -march=native -DNDEBUG "$C_SRC" -o "$BIN_DIR/bench_c"

build_ark_source_for_n "$ARK_SRC_TEMPLATE" "$ARK_SRC_GEN" "$N"

# Ark: build raw/bare to avoid capsule sealing overhead during benchmark
"$ARKNET_BIN" compile --bare -o "$BIN_DIR/bench_ark" "$ARK_SRC_GEN"

echo
echo "== Sanity (checksums) =="

c_out="$("$BIN_DIR/bench_c" "$N")"
py_out="$(python3 "$PY_SRC" "$N")"
ark_out="$("$BIN_DIR/bench_ark")"

printf '%s\n' "$c_out"
printf '%s\n' "$py_out"
printf '%s\n' "$ark_out"

c_sum="$(extract_checksum "$c_out")"   || { echo "[ERROR] Failed to parse C checksum" >&2; exit 1; }
py_sum="$(extract_checksum "$py_out")" || { echo "[ERROR] Failed to parse Python checksum" >&2; exit 1; }
ark_sum="$(extract_checksum "$ark_out")" || { echo "[ERROR] Failed to parse Ark checksum" >&2; exit 1; }

if [[ "$c_sum" != "$py_sum" || "$c_sum" != "$ark_sum" ]]; then
  echo "[ERROR] Checksum mismatch: c=$c_sum py=$py_sum ark=$ark_sum" >&2
  exit 1
fi

echo
echo "== Timing (N=$N, runs=$RUNS) =="

for ((r = 1; r <= RUNS; ++r)); do
  echo
  echo "-- Run $r / C --"
  time_cmd "$BIN_DIR/bench_c" "$N" >/dev/null

  echo "-- Run $r / Python --"
  time_cmd python3 "$PY_SRC" "$N" >/dev/null

  echo "-- Run $r / Ark --"
  time_cmd "$BIN_DIR/bench_ark" >/dev/null
done