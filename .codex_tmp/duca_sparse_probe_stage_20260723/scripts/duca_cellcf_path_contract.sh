#!/usr/bin/env bash

duca_cellcf_require_external_path() {
  local label="$1"
  local repo_root="$2"
  local base_root="$3"
  local requested_path="$4"
  local repo_real
  local base_real
  local target_real

  [[ -n "${requested_path}" ]] || {
    echo "[DUCA_CELLCF_PATH][FAIL] ${label} is required" >&2
    return 1
  }
  command -v realpath >/dev/null 2>&1 || {
    echo "[DUCA_CELLCF_PATH][FAIL] realpath is required" >&2
    return 1
  }
  repo_real="$(realpath -e -- "${repo_root}")" || return 1
  base_real="$(realpath -e -- "${base_root}")" || {
    echo "[DUCA_CELLCF_PATH][FAIL] BASE does not exist: ${base_root}" >&2
    return 1
  }
  target_real="$(realpath -m -- "${requested_path}")" || return 1
  [[ "${target_real}" != "${base_real}" ]] || {
    echo "[DUCA_CELLCF_PATH][FAIL] ${label} cannot equal BASE" >&2
    return 1
  }
  case "${target_real}/" in
    "${base_real}/"*) ;;
    *)
      echo "[DUCA_CELLCF_PATH][FAIL] ${label} must stay under BASE" >&2
      return 1
      ;;
  esac
  case "${target_real}/" in
    "${repo_real}/"*)
      echo "[DUCA_CELLCF_PATH][FAIL] ${label} must stay outside the worktree" >&2
      return 1
      ;;
  esac
  printf '%s\n' "${target_real}"
}
