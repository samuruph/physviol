#!/usr/bin/env bash
# Walk the whole compatibility matrix, then build every video worth looking at.
#
#   bash scripts/generate_sample.sh [complexity] [tier] [seed] [severity] [window] [variants]
#
# Defaults to ONE `strong` variant per cell at L0/Tier D -- the fastest way to
# see all 177 cells. While checking that scenarios and families work at all,
# breadth of coverage is what you want to look at, not three strengths of the
# same thing; pass `all` as the severity for the full ladder.
#
# `window` defaults to EMPTY, i.e. the CLI is left to size each violation as a
# fraction of the clip (`_geom.WINDOW_FRACTION`). It used to default to 4 frames,
# which was fine at 13-frame clips and became a bug at 25 and 49: a uniform
# 4-frame window makes every sustained family a brief blip in a long clip, and
# it silently overrode `permanence`'s strong bin, which is supposed to mean
# "gone for good". Pass a number here only when you deliberately want every
# family pinned to the same duration.
#
# The CLI walks the matrix itself and groups every family of a scenario into a
# single container run, so this script is a thin wrapper around one command
# plus the videos.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CX="${1:-L0}"; TIER="${2:-D}"; SEED="${3:-777}"
SEV="${4:-strong}"; WIN="${5:-}"; VARIANTS="${6:-1}"
REL=out/release

echo "== generating: $CX / tier $TIER / severity $SEV / window ${WIN:-scaled} / $VARIANTS variant(s) =="
conda run --no-capture-output -n physviol python -m physviol.cli generate \
    --tier "$TIER" --complexity "$CX" --seed "$SEED" --severity "$SEV" \
    ${WIN:+--window "$WIN"} --variants "$VARIANTS" --keep-going \
    --workdir out/work --outdir "$REL"

echo "== grids: the valid clip beside every severity of each family =="
for PAIR in $(find "$REL/clips" -mindepth 3 -maxdepth 3 -type d | sort); do
  FAMS=$(ls "$PAIR" | sed -n 's/^invalid_\(.*\)_\(weak\|medium\|strong\)$/\1/p' | sort -u)
  for FAM in $FAMS; do
    conda run -n physviol python -m physviol.cli grid "$PAIR" --family "$FAM" 2>/dev/null \
      | grep -oE '"path": "[^"]*"' | sed 's/"path": /  /'
  done
done

echo "== coverage: every invalid clip in one video =="
conda run -n physviol python -m physviol.cli coverage "$REL" \
  | grep -oE '"path": "[^"]*"' | sed 's/"path": /  /'

conda run -n physviol python -m physviol.cli validate "$REL"
