#!/usr/bin/env bash
# Generate a release from a config, validate it, and build everything worth
# looking at.
#
#   bash scripts/run.sh                # the review sweep (configs/review.yaml)
#   bash scripts/run.sh v0_release
#   bash scripts/run.sh review --tier v0     # extra flags pass straight through
#
# The config decides tier, complexity, severity, seed and variants; see
# configs/*.yaml, which document every key. Anything after the config name is
# forwarded to `generate` and overrides the file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${1:-review}"
shift || true
PV="conda run --no-capture-output -n physviol python -m physviol.cli"

echo "== generate: --config $CONFIG $* =="
$PV generate --config "$CONFIG" "$@"

REL=$($PV config-path --config "$CONFIG" 2>/dev/null || echo "out/release")

echo "== validate =="
$PV validate "$REL" || true

echo "== coverage: every invalid clip in one video =="
$PV coverage "$REL" || true

echo "== sheets: every family of a scenario at once, per view =="
for PAIR in $(find "$REL/clips" -mindepth 3 -maxdepth 3 -type d | sort); do
  for VIEW in mask energy; do
    $PV sheet "$PAIR" --view "$VIEW" || true
  done
done

echo "== grids: the valid clip beside every severity of each family =="
for PAIR in $(find "$REL/clips" -mindepth 3 -maxdepth 3 -type d | sort); do
  FAMS=$(ls "$PAIR" | sed -n 's/^invalid_\(.*\)_\(weak\|medium\|strong\)$/\1/p' | sort -u)
  for FAM in $FAMS; do
    $PV grid "$PAIR" --family "$FAM" || true
  done
done

echo
echo "done -> $REL"
echo "  coverage.mp4        every cell in one video -- open this first"
echo "  clips/*/*/*/sheet_mask.mp4    every family of a scenario, side by side"
echo "  clips/*/*/*/grid_<family>.mp4 one family, every severity, every view"
