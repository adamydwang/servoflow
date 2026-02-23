#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ServoFlow vs HuggingFace RDT-1B comparison benchmark.
#
# Usage:
#   bash benchmarks/run_comparison.sh [--steps N] [--iters K] [--dtype fp16]
#
# Requires:
#   - 'servoflow:latest' image  (docker-build.sh)
#   - 'servoflow-bench:latest'  (docker build -f Dockerfile.bench ...)
#   - GPU available (--gpus all)
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
STEPS=10
ITERS=20
WARMUP=5
DTYPE="fp16"
ACTION_DIM=14
ACTION_HORIZON=64
GPU="${GPU:-all}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODEL_ID="robotics-diffusion-transformer/rdt-1b"

while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)   STEPS="$2";   shift 2 ;;
        --iters)   ITERS="$2";   shift 2 ;;
        --warmup)  WARMUP="$2";  shift 2 ;;
        --dtype)   DTYPE="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BENCH_IMAGE="servoflow-bench:latest"
SF_IMAGE="servoflow:latest"
HF_JSON="/tmp/sf_bench_hf_$$.json"
SF_JSON="/tmp/sf_bench_sf_$$.json"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       ServoFlow vs HuggingFace RDT-1B Benchmark              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  denoising steps : %-42s ║\n" "$STEPS"
printf "║  measure iters   : %-42s ║\n" "$ITERS"
printf "║  dtype           : %-42s ║\n" "$DTYPE"
printf "║  action dim×hor  : %-42s ║\n" "${ACTION_DIM}×${ACTION_HORIZON}"
printf "║  model           : %-42s ║\n" "$MODEL_ID"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Build bench image if not present ─────────────────────────────────────────
if ! docker image inspect "$BENCH_IMAGE" &>/dev/null; then
    echo "► Building $BENCH_IMAGE …"
    docker build \
        -f "$(dirname "$0")/../Dockerfile.bench" \
        -t "$BENCH_IMAGE" \
        "$(dirname "$0")/.."
fi

# ── 1. HuggingFace baseline ───────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [1/2]  HuggingFace Transformers — RDT-1B"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker run --rm --gpus "device=${GPU}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${HF_JSON}:${HF_JSON}" \
    "$BENCH_IMAGE" \
    python benchmarks/bench_hf_rdt1b.py \
        --steps          "$STEPS" \
        --iters          "$ITERS" \
        --warmup         "$WARMUP" \
        --dtype          "$DTYPE" \
        --action-dim     "$ACTION_DIM" \
        --action-horizon "$ACTION_HORIZON" \
        --model-id       "$MODEL_ID" \
        --output-json    "$HF_JSON" \
    || true   # non-fatal if model download fails

echo ""

# ── 2. ServoFlow pipeline benchmark ──────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [2/2]  ServoFlow (framework overhead, stub RDT-1B dimensions)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker run --rm --gpus "device=${GPU}" \
    "$SF_IMAGE" \
    /workspace/servoflow/build/benchmarks/bench_pipeline "$STEPS"

echo ""

# ── 3. Summary table ──────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    COMPARISON SUMMARY                        ║"
echo "╠════════════════════════╦══════════════╦══════════════════════╣"
echo "║ Metric                 ║  HuggingFace ║  ServoFlow (stub)    ║"
echo "╠════════════════════════╬══════════════╬══════════════════════╣"

if [[ -f "$HF_JSON" ]]; then
    # Parse key metrics from HF JSON using python
    python3 - "$HF_JSON" "$STEPS" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    r = json.load(f)
steps = int(sys.argv[2])

def get_ms(r, keys):
    for k in keys:
        if k in r:
            return f"{r[k]['mean_ms']:.1f} ms"
    return "  N/A"

full  = get_ms(r, ["full_pipeline",      "dummy_single_forward"])
loop  = get_ms(r, ["denoise_loop",       "dummy_denoise_loop"])
step  = f"{r.get('per_step_ms', 0):.2f} ms" if 'per_step_ms' in r else "N/A"
hz    = f"{1000/(r.get('per_step_ms',1)*steps):.1f} Hz" if 'per_step_ms' in r else "N/A"
mem   = f"{r.get('model_memory_mb',0):.0f} MB" if 'model_memory_mb' in r else "N/A"
vmem  = f"{r.get('peak_memory_mb',0):.0f} MB"  if 'peak_memory_mb'  in r else "N/A"

print(f"║ Single forward pass    ║ {full:>12} ║ (see ServoFlow output) ║")
print(f"║ {steps}-step denoise loop  ║ {loop:>12} ║ (see ServoFlow output) ║")
print(f"║ Per-step latency       ║ {step:>12} ║ (see ServoFlow output) ║")
print(f"║ Achievable control Hz  ║ {hz:>12} ║ (see ServoFlow output) ║")
print(f"║ Model VRAM             ║ {mem:>12} ║ ~framework only        ║")
print(f"║ Peak VRAM              ║ {vmem:>12} ║ (see ServoFlow output) ║")
PYEOF
    rm -f "$HF_JSON"
else
    echo "║ (HuggingFace results not available)                          ║"
fi

echo "╠════════════════════════╩══════════════╩══════════════════════╣"
echo "║ NOTE: ServoFlow stub uses same tensor dimensions as RDT-1B.  ║"
echo "║ Real model comparison pending weight loader completion.       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
