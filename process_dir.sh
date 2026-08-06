#!/usr/bin/env bash
# process_dir.sh <src_dir> <dst_dir>
#
# Run the Masala-CHAI V1 pipeline on every image under <src_dir> and drop all
# artifacts under <dst_dir>. Assumes:
#   - ~/venv exists with the packages listed in
#     ~/gdrive/claude/01_projects/schematics/masala_chai.md
#   - A local OpenAI-compatible vision LLM is running at
#     $OPENAI_API_BASE (default http://127.0.0.1:8000/v1) serving
#     $MASALA_MODEL (default Qwen/Qwen3-VL-8B-Instruct).
#   - hough/checkpoint.pth and hough/htlcnn/ are populated.
#
# Example:
#   ./process_dir.sh ~/fig_in ~/fig_out

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <src_dir> <dst_dir>" >&2
    exit 2
fi

SRC_DIR="$1"
DST_DIR="$2"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "error: <src_dir> '$SRC_DIR' is not a directory" >&2
    exit 1
fi

mkdir -p "$DST_DIR"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Localhost LLM — do NOT go through fwdproxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Override any pre-set OPENAI_* env (e.g. hackathon Llama endpoint in the
# user's shell) unless the caller explicitly opts out with MASALA_KEEP_ENV=1.
if [[ "${MASALA_KEEP_ENV:-0}" != "1" ]]; then
    export OPENAI_API_BASE="${MASALA_LLM_BASE:-http://127.0.0.1:8000/v1}"
    export MASALA_MODEL="${MASALA_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
    export OPENAI_API_KEY="none"
else
    : "${OPENAI_API_BASE:=http://127.0.0.1:8000/v1}"
    : "${MASALA_MODEL:=Qwen/Qwen3-VL-8B-Instruct}"
    : "${OPENAI_API_KEY:=none}"
    export OPENAI_API_BASE MASALA_MODEL OPENAI_API_KEY
fi

if [[ -f "$HOME/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/venv/bin/activate"
fi

echo "[process_dir] repo=$REPO_DIR"
echo "[process_dir] src=$SRC_DIR"
echo "[process_dir] dst=$DST_DIR"
echo "[process_dir] llm=$OPENAI_API_BASE model=$MASALA_MODEL"

python run.py --src "$SRC_DIR" --tgt "$DST_DIR" --api_key "$OPENAI_API_KEY"

echo "[process_dir] done. artifacts under: $DST_DIR"
