#!/usr/bin/env bash
# Idempotent: clones ik_llama.cpp, checks out the verified base SHA, applies
# PR #1744 + pestopoppa's two follow-up fixes, leaves you with a clean tree
# ready to build.
#
# Usage:
#   ./scripts/apply_patches.sh              # uses ./build/ik_llama.cpp
#   ./scripts/apply_patches.sh /path/to/dir # uses that path

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-${REPO_ROOT}/build/ik_llama.cpp}"
BASE_SHA="9895026"   # verified ik_llama.cpp HEAD that PR 1744 sits cleanly on

log() { printf '\033[1;34m[apply_patches]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[apply_patches] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. Clone or reuse
if [[ -d "$TARGET/.git" ]]; then
  log "reusing existing clone at $TARGET"
  cd "$TARGET"
  git fetch --quiet origin
  git checkout --quiet "$BASE_SHA"
  git clean -fdx --quiet
else
  log "cloning ik_llama.cpp into $TARGET"
  mkdir -p "$(dirname "$TARGET")"
  git clone --quiet https://github.com/ikawrakow/ik_llama.cpp.git "$TARGET"
  cd "$TARGET"
  git checkout --quiet "$BASE_SHA"
fi

log "base HEAD: $(git rev-parse --short HEAD) — $(git log -1 --format=%s)"

# 2. Apply the consolidated PR 1744 diff
log "applying patches/0001-PR-1744-gemma4-mtp.patch"
if git apply --check "${REPO_ROOT}/patches/0001-PR-1744-gemma4-mtp.patch" 2>/dev/null; then
  git apply "${REPO_ROOT}/patches/0001-PR-1744-gemma4-mtp.patch"
  git add -A && git -c user.email=apply@local -c user.name=apply commit -q -m "Apply ik_llama.cpp PR #1744 (Gemma 4 MTP)"
else
  err "PR 1744 patch failed to apply cleanly. Likely the base SHA drifted; consider 'git fetch origin pull/1744/head:pr-1744' and merge directly."
fi

# 3. Apply pestopoppa's segfault fix via in-place edit (more robust than a unified diff).
#    Note: at the verified base SHA 9895026, PR 1744 itself already contains this fix
#    (pestopoppa's PR-comment fix got merged into the PR). The needle check below is
#    a true precondition: if the buggy line is absent, the fix is already in and we skip.
#    The patch remains as a defensive backup against upstream regression.
log "applying pestopoppa fix #1: params_use_gemma4_external_mtp segfault"
SERVER_CTX="examples/server/server-context.cpp"
if grep -q 'params_base.speculative.type == COMMON_SPECULATIVE_TYPE_MTP &&' "$SERVER_CTX"; then
  # Remove the line; this is a multi-line condition so we use sed
  python3 - <<'PY'
import re, sys
p = "examples/server/server-context.cpp"
src = open(p).read()
needle = "        params_base.speculative.type == COMMON_SPECULATIVE_TYPE_MTP &&\n"
if needle in src:
    src = src.replace(needle, "")
    open(p, "w").write(src)
    print("    -> removed chicken-and-egg precondition")
else:
    print("    -> needle not found; PR 1744 base may have shifted")
    sys.exit(1)
PY
  git add "$SERVER_CTX"
  git -c user.email=apply@local -c user.name=apply commit -q -m "Fix params_use_gemma4_external_mtp chicken-and-egg (pestopoppa, PR 1744 comment)"
else
  log "    -> already applied or PR 1744 base shifted; skipping"
fi

# 4. Apply pestopoppa's tensor-name-warning silence.
#    Same caveat as fix #1 — already in PR 1744 at base SHA 9895026. The needle check
#    below detects whether the fixed block is already present and skips re-applying.
log "applying pestopoppa fix #2: silence MTP tensor-name warnings"
LLAMA_CPP="src/llama.cpp"
if ! grep -q 'mtp_pre_proj.weight.*mtp_post_proj.weight' "$LLAMA_CPP"; then
  python3 - <<'PY'
import sys
p = "src/llama.cpp"
src = open(p).read()
needle = '''        if (name == "output_norm.weight") {
            continue;
        }
        auto pos = name.find("blk.");'''
patch = '''        if (name == "output_norm.weight") {
            continue;
        }
        // Top-level (non-blk.*) tensors introduced by Gemma4Assistant (gemma4_mtp arch).
        // Loaded by create_gemma4_mtp_tensors; not per-layer, so they don't participate in this accounting.
        if (name == "mtp_pre_proj.weight" || name == "mtp_post_proj.weight" ||
            name == "mtp_centroids.weight" || name == "mtp_token_ordering.weight") {
            continue;
        }
        auto pos = name.find("blk.");'''
if needle in src:
    src = src.replace(needle, patch)
    open(p, "w").write(src)
    print("    -> warning-silence inserted")
else:
    print("    -> needle not found in src/llama.cpp; PR 1744 may have shifted the surrounding code")
    sys.exit(1)
PY
  git add "$LLAMA_CPP"
  git -c user.email=apply@local -c user.name=apply commit -q -m "Silence Oops: tensor with strange name mtp_*.weight warnings (pestopoppa)"
else
  log "    -> already applied; skipping"
fi

log "done — ik_llama.cpp tree ready at $TARGET"
log "branch:   $(git rev-parse --abbrev-ref HEAD)"
log "head:     $(git rev-parse --short HEAD)"
log "next:     ./scripts/build_cuda_linux.sh   (or build_cuda_windows.ps1)"
