#!/usr/bin/env bash
# Task 2: SONIC v1.1 with planner-idle / VR_3PT, using verified local LFS data.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR=/home/drgh/g1_official/GR00T-WholeBodyControl/gear_sonic_deploy
REF_DIR="$PROJECT_DIR/sonic_reference"
BINARY="$DEPLOY_DIR/target/release/g1_deploy_onnx_ref"

if [[ "${1:-}" != --check && "${1:-}" != --real ]]; then
    echo "Usage: bash $0 --check|--real" >&2
    exit 2
fi

[[ -x "$BINARY" ]] || { echo "SONIC binary missing: $BINARY" >&2; exit 1; }
[[ -d "$REF_DIR" ]] || { echo "Verified motion references missing: $REF_DIR" >&2; exit 1; }
for name in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
    [[ -s "$DEPLOY_DIR/policy/sonic_v1_1/$name" ]] || {
        echo "Missing SONIC v1.1 file: $name" >&2; exit 1;
    }
done
[[ -s "$DEPLOY_DIR/planner/target_vel/V2/planner_sonic.onnx" ]] || {
    echo "Planner ONNX missing" >&2; exit 1;
}
[[ -d /home/drgh/NVIDIA/TensorRT/lib ]] || {
    echo "TensorRT library directory missing" >&2; exit 1;
}
/usr/bin/python3 "$PROJECT_DIR/tools/verify_sonic_refs.py" \
    --repo /home/drgh/g1_official/GR00T-WholeBodyControl \
    --reference-dir "$REF_DIR"
echo "Verified task-2 SONIC references: 78 real CSVs in $REF_DIR"
echo "SONIC binary/model/planner/TensorRT: present"

if [[ "$1" == --check ]]; then
    echo "Check only; no robot commands sent."
    exit 0
fi

export TensorRT_ROOT=/home/drgh/NVIDIA/TensorRT
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:${LD_LIBRARY_PATH:-}"
export FASTRTPS_DEFAULT_PROFILES_FILE="$DEPLOY_DIR/src/g1/g1_deploy_onnx_ref/config/fastrtps_profile.xml"
cd "$DEPLOY_DIR"
exec "$BINARY" enp8s0 \
    policy/sonic_v1_1/model_decoder.onnx "$REF_DIR" \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --encoder-file policy/sonic_v1_1/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type zmq --zmq-host localhost
