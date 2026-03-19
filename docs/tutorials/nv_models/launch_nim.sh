#!/usr/bin/env bash
set -euo pipefail

DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/ephemeral/docker-data}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] IMAGE

Launch a NIM container in the background.

Commands:
  setup-docker              Move Docker data-root to \$DOCKER_DATA_ROOT (run once)

Required:
  IMAGE                     Docker image (e.g. nvcr.io/nim/nvidia/nemotron-parse:latest)

Options:
  -n, --name   NAME         Container name                       (required)
  -g, --gpus   GPUS         GPU device(s): 0, "1,2", or "all"   (required)
  -p, --port   PORT         Host port to map to container 8000   (required)
  -l, --log    FILE         Log file path                        (default: <name>.log)
  -s, --shm    SIZE         Shared memory size                   (default: 16GB)
  -c, --cache  DIR          Local NIM cache directory             (default: ~/ephemeral/.cache/nim)
  -d, --docker-root DIR     Docker data-root for setup-docker    (default: /ephemeral/docker-data)
  -h, --help                Show this help message
EOF
    exit 1
}

# ── setup-docker: move Docker data-root to ephemeral storage ─────────
setup_docker() {
    local ephemeral_root
    ephemeral_root="$(dirname "$DOCKER_DATA_ROOT")"

    # Ensure /tmp exists and points to ephemeral storage (root disk is often too small)
    if ! mountpoint -q /tmp 2>/dev/null || [[ "$(df --output=target /tmp 2>/dev/null | tail -1)" != "$ephemeral_root"* ]]; then
        echo "==> Binding /tmp → $ephemeral_root/tmp (avoids filling root disk)"
        sudo mkdir -p "$ephemeral_root/tmp" /tmp
        sudo chmod 1777 "$ephemeral_root/tmp" /tmp
        sudo mount --bind "$ephemeral_root/tmp" /tmp
    fi

    local containerd_root="$ephemeral_root/containerd"
    local containerd_cfg="/etc/containerd/config.toml"
    local needs_restart=false

    # Move containerd root to ephemeral
    if [[ -f "$containerd_cfg" ]] && ! grep -q "root.*=.*\"$containerd_root\"" "$containerd_cfg"; then
        echo "==> Moving containerd root → $containerd_root"
        sudo mkdir -p "$containerd_root"
        sudo sed -i "s|^#*root\s*=.*|root = \"$containerd_root\"|" "$containerd_cfg"
        needs_restart=true
    fi

    # Move Docker data-root to ephemeral
    echo "==> Configuring Docker data-root → $DOCKER_DATA_ROOT"
    sudo mkdir -p "$DOCKER_DATA_ROOT"

    local daemon_json="/etc/docker/daemon.json"
    if ! [[ -f "$daemon_json" ]] || ! grep -q "$DOCKER_DATA_ROOT" "$daemon_json"; then
        local existing="{}"
        [[ -f "$daemon_json" ]] && existing=$(cat "$daemon_json")

        echo "$existing" | python3 -c "
import sys, json
cfg = json.load(sys.stdin)
cfg['data-root'] = '$DOCKER_DATA_ROOT'
json.dump(cfg, sys.stdout, indent=4)
print()
" | sudo tee "$daemon_json" > /dev/null
        needs_restart=true
    fi

    if $needs_restart; then
        echo "    Restarting containerd + Docker..."
        sudo systemctl restart containerd
        sudo systemctl restart docker
    fi

    echo "    Docker Root Dir: $(docker info 2>/dev/null | grep 'Docker Root Dir' | awk '{print $NF}')"
    echo "    containerd root: $(sudo containerd config dump 2>/dev/null | grep "^root" | awk -F"'" '{print $2}')"

    for d in /var/lib/docker /var/lib/containerd; do
        if [[ -d "$d" ]]; then
            echo "    Removing old $d to reclaim space on /..."
            sudo rm -rf "$d"
        fi
    done
    echo "    Root disk: $(df -h / | awk 'NR==2{print $4, "free"}')"
}

NAME="" GPUS="" PORT="" LOG="" SHM="16GB" CACHE="${LOCAL_NIM_CACHE:-$HOME/ephemeral/.cache/nim}"
SUBCOMMAND=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        setup-docker)     SUBCOMMAND="setup-docker"; shift ;;
        -n|--name)        NAME="$2";  shift 2 ;;
        -g|--gpus)        GPUS="$2";  shift 2 ;;
        -p|--port)        PORT="$2";  shift 2 ;;
        -l|--log)         LOG="$2";   shift 2 ;;
        -s|--shm)         SHM="$2";   shift 2 ;;
        -c|--cache)       CACHE="$2";           shift 2 ;;
        -d|--docker-root) DOCKER_DATA_ROOT="$2"; shift 2 ;;
        -h|--help)        usage ;;
        -*)               echo "Unknown option: $1" >&2; usage ;;
        *)                IMAGE="$1"; shift ;;
    esac
done

if [[ "$SUBCOMMAND" == "setup-docker" ]]; then
    setup_docker
    exit 0
fi

[[ -z "${NAME:-}" ]] && { echo "Error: --name is required" >&2; usage; }
[[ -z "${GPUS:-}" ]] && { echo "Error: --gpus is required" >&2; usage; }
[[ -z "${PORT:-}" ]] && { echo "Error: --port is required" >&2; usage; }
[[ -z "${IMAGE:-}" ]] && { echo "Error: IMAGE argument is required" >&2; usage; }

: "${NGC_API_KEY:?NGC_API_KEY must be set}"

LOG="${LOG:-${NAME}.log}"
mkdir -p "$CACHE"

echo "Launching NIM container '$NAME' on GPU(s) $GPUS → localhost:$PORT"
echo "  Image: $IMAGE"
echo "  Log:   $LOG"

CONTAINER_TMP="${CACHE}/tmp-${NAME}"
mkdir -p "$CONTAINER_TMP"

if [[ "$GPUS" == "all" ]]; then
    GPU_ARG="all"
else
    GPU_ARG="\"device=${GPUS}\""
fi

nohup docker run --rm --name "$NAME" \
    --gpus "$GPU_ARG" \
    --shm-size="$SHM" \
    -e NGC_API_KEY \
    -e TMPDIR=/tmp \
    -v "$CACHE:/opt/nim/.cache" \
    -v "$CONTAINER_TMP:/tmp" \
    -u 0:0 \
    -p "${PORT}:8000" \
    "$IMAGE" > "$LOG" 2>&1 &

echo "Container started (PID $!). Tailing log — Ctrl+C to stop tailing (container keeps running)."
tail -f "$LOG"
