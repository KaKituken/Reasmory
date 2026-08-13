#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="."
DEFAULT_RRD="./debug_vis/04_13_spatial_memory_counting.rrd"
DEFAULT_PORT="9090"

usage() {
    echo "Usage: $0 [path_to_rrd] [--native] [--port PORT]"
    echo
    echo "Default mode uses the browser viewer, which works on headless servers."
    echo "Examples:"
    echo "  $0"
    echo "  $0 /path/to/file.rrd"
    echo "  $0 --port 9091"
    echo "  $0 /path/to/file.rrd --native"
}

RRD_PATH="${DEFAULT_RRD}"
PORT="${DEFAULT_PORT}"
USE_NATIVE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --native)
            USE_NATIVE=1
            shift
            ;;
        --port)
            PORT="${2:-}"
            if [[ -z "${PORT}" ]]; then
                usage
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ "${RRD_PATH}" != "${DEFAULT_RRD}" ]]; then
                usage
                exit 1
            fi
            RRD_PATH="$1"
            shift
            ;;
    esac
done

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen

cd "${ROOT_DIR}"

if [[ "${USE_NATIVE}" -eq 1 ]]; then
    echo "Opening native Rerun viewer:"
    echo "  ${RRD_PATH}"
    rerun "${RRD_PATH}"
    exit 0
fi

echo "Starting Rerun web viewer for:"
echo "  ${RRD_PATH}"
echo
echo "Open this URL in your browser:"
echo "  http://$(hostname -f):${PORT}"
echo
echo "If you are on a remote machine, port-forward first, for example:"
echo "  ssh -L ${PORT}:localhost:${PORT} $(whoami)@$(hostname -f)"

rerun "${RRD_PATH}" --web-viewer --web-viewer-port "${PORT}"
