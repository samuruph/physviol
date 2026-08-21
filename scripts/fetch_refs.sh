#!/usr/bin/env bash
# Fetch read-only reference checkouts into refs/ (gitignored).
#
# These are NOT dependencies. Kubric is already installed inside the docker image
# we render with; this clone exists so the source is greppable and so the MOVi
# worker templates can be read locally. See docs/PLAN.md Part 0 and Part 0.5.
#
# The image's installed Kubric is the API authority — this pin may drift from it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFS_DIR="$REPO_ROOT/refs"

KUBRIC_URL="https://github.com/google-research/kubric.git"
KUBRIC_SHA="61f2422c84bab75006df33c6989e0b483db3ccfe"
KUBRIC_DIR="$REFS_DIR/kubric"

mkdir -p "$REFS_DIR"

if [ -d "$KUBRIC_DIR/.git" ]; then
  current="$(git -C "$KUBRIC_DIR" rev-parse HEAD 2>/dev/null || echo none)"
  if [ "$current" = "$KUBRIC_SHA" ]; then
    echo "kubric: already at pinned $KUBRIC_SHA"
    exit 0
  fi
  echo "kubric: at $current, moving to pinned $KUBRIC_SHA"
else
  echo "kubric: initializing $KUBRIC_DIR"
  rm -rf "$KUBRIC_DIR"
  mkdir -p "$KUBRIC_DIR"
  git -C "$KUBRIC_DIR" init -q
  git -C "$KUBRIC_DIR" remote add origin "$KUBRIC_URL"
fi

# Shallow fetch of exactly the pinned commit; no history, no other branches.
git -C "$KUBRIC_DIR" fetch -q --depth 1 origin "$KUBRIC_SHA"
git -C "$KUBRIC_DIR" checkout -q FETCH_HEAD

echo "kubric: $(git -C "$KUBRIC_DIR" rev-parse HEAD)"
echo
echo "Start here:"
echo "  refs/kubric/challenges/movi/movi_def_worker.py   <- the template (Part 0.5)"
echo "  refs/kubric/examples/                            <- smaller, simpler workers"
echo "  refs/kubric/kubric/renderer/blender.py           <- what the exporters actually emit"
