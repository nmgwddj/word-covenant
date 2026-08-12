#!/usr/bin/env bash

set -euo pipefail

if [ "${1:-}" = '--' ]; then
  shift
fi

if [ "$#" -ne 1 ]; then
  printf 'Usage: %s <WordCovenant.dmg>\n' "$0" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dmg_path="$1"

if [ ! -f "${dmg_path}" ]; then
  printf 'DMG does not exist: %s\n' "${dmg_path}" >&2
  exit 1
fi

mount_path="$(mktemp -d "${TMPDIR:-/tmp}/word-covenant-dmg.XXXXXX")"
mounted=0

cleanup() {
  if [ "${mounted}" -eq 1 ]; then
    hdiutil detach "${mount_path}" -quiet >/dev/null 2>&1 || true
  fi
  rmdir "${mount_path}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

hdiutil verify "${dmg_path}"
hdiutil attach "${dmg_path}" -readonly -nobrowse -mountpoint "${mount_path}" >/dev/null
mounted=1

app_path="${mount_path}/WordCovenant.app"
resource_root="${app_path}/Contents/Resources"
model_root="${resource_root}/models"

test -d "${app_path}"
test "$(plutil -extract LSMinimumSystemVersion raw "${app_path}/Contents/Info.plist")" = '11.0'
test -s "${resource_root}/third-party/whisper-large-v3-turbo-model-card.txt"
test -s "${resource_root}/third-party/whisper-large-v3-turbo-model-MIT.txt"
test -f "${model_root}/ggml-large-v3-turbo-q5_0.bin"
test -f "${model_root}/manifest.json"
test "$(find "${model_root}" -mindepth 1 -maxdepth 1 -type f -print | wc -l | tr -d '[:space:]')" = '2'
test "$(find "${model_root}" -mindepth 1 -maxdepth 1 ! -type f -print | wc -l | tr -d '[:space:]')" = '0'
test "$(stat -f '%Lp' "${model_root}/ggml-large-v3-turbo-q5_0.bin")" = '644'
cmp -s "${project_root}/models/whisper-large-v3-turbo-q5_0.lock.json" "${model_root}/manifest.json"

cd "${project_root}"
pnpm model:verify -- --manifest "${model_root}/manifest.json" --destination "${model_root}/ggml-large-v3-turbo-q5_0.bin"

printf 'Verified packaged WordCovenant DMG: %s\n' "${dmg_path}"
