#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: fetch_pvg_models.sh [--repo-type TYPE] [--revision REV] [--cache-root DIR] MODEL_ID...

Downloads Hugging Face model weights into the Isambard-friendly cache location.

Options:
  --repo-type TYPE   Repo type (default: model)
  --revision REV     Optional revision/commit/branch
  --cache-root DIR   Override large-storage root (defaults to PROJECTDIR/SCRATCHDIR/HOME)
  -h, --help         Show this help

Examples:
  ./fetch_pvg_models.sh Qwen/Qwen2.5-0.5B-Instruct
  ./fetch_pvg_models.sh --revision main meta-llama/Llama-3.1-8B
  LUDIC_LARGE_STORAGE=/path/to/storage ./fetch_pvg_models.sh Qwen/Qwen2.5-7B
USAGE
}

repo_type="model"
revision=""
cache_root=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-type)
      repo_type="${2:-}"
      shift 2
      ;;
    --revision)
      revision="${2:-}"
      shift 2
      ;;
    --cache-root)
      cache_root="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

resolve_cache_root() {
  if [[ -n "${LUDIC_LARGE_STORAGE:-}" ]]; then
    printf '%s\n' "$LUDIC_LARGE_STORAGE"
    return 0
  fi
  if [[ -n "${LUDIC_PROJECTDIR:-}" ]]; then
    printf '%s\n' "$LUDIC_PROJECTDIR"
    return 0
  fi
  if [[ -n "${PROJECTDIR:-}" ]]; then
    printf '%s\n' "$PROJECTDIR"
    return 0
  fi
  if [[ -n "${SCRATCHDIR:-}" ]]; then
    printf '%s\n' "$SCRATCHDIR"
    return 0
  fi
  if [[ -n "${SCRATCH:-}" ]]; then
    printf '%s\n' "$SCRATCH"
    return 0
  fi
  printf '%s\n' "$HOME"
}

if [[ -z "$cache_root" ]]; then
  cache_root="$(resolve_cache_root)"
fi

HF_HOME="${cache_root%/}/.cache/huggingface"
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"

mkdir -p "$HF_HUB_CACHE"

download_with_cli() {
  local repo_id="$1"
  shift
  local cmd=("$@" "$repo_id" --repo-type "$repo_type" --cache-dir "$HF_HUB_CACHE")
  if [[ -n "$revision" ]]; then
    cmd+=(--revision "$revision")
  fi
  "${cmd[@]}"
}

download_with_python() {
  local repo_id="$1"
  python - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${repo_id}",
    repo_type="${repo_type}",
    revision="${revision}" if "${revision}" else None,
    cache_dir=os.environ["HF_HUB_CACHE"],
)
PY
}

for repo_id in "$@"; do
  echo "==> Downloading ${repo_id} into ${HF_HUB_CACHE}"
  if command -v hf >/dev/null 2>&1; then
    download_with_cli "$repo_id" hf download
  elif command -v huggingface-cli >/dev/null 2>&1; then
    download_with_cli "$repo_id" huggingface-cli download
  else
    download_with_python "$repo_id"
  fi
done

echo "Done. Cache location: ${HF_HUB_CACHE}"
