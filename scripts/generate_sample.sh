#!/usr/bin/env bash
# Generate one batch per runnable (scenario, family) cell -- all three severity
# bins in a single container run each -- then build a comparison grid per pair.
#
#   bash scripts/generate_sample.sh [complexity] [tier] [seed] [severity]
#
# Defaults to ONE `strong` variant per cell. While checking that scenarios and
# families work at all, breadth of coverage is what you want to look at, not
# three strengths of the same thing -- pass `all` for the full ladder.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CX="${1:-L1}"; TIER="${2:-D}"; SEED="${3:-777}"; SEV="${4:-strong}"; WIN="${5:-4}"
WORK=out/work; REL=out/release

CELLS=$(conda run -n physviol python -c "
import sys; sys.path.insert(0,'.')
from physviol import scenarios, injectors
from physviol.taxonomy import build_cells
hs=set(scenarios.available()); hi=set(injectors.available())
print('\n'.join('%s %s'%c for c in build_cells() if c[0] in hs and c[1] in hi))
" 2>/dev/null | grep -v '^$')

echo "== generating $(echo "$CELLS" | wc -l) cells [severity=$SEV, window=$WIN, $CX, tier $TIER] =="
START=$(date +%s)
while read -r SCEN FAM; do
  [ -z "$SCEN" ] && continue
  conda run -n physviol python -m physviol.cli generate \
      --tier "$TIER" -n 1 --seed "$SEED" --scenario "$SCEN" --family "$FAM" \
      --complexity "$CX" --severity "$SEV" --window "$WIN" \
      --workdir "$WORK" --outdir "$REL" 2>&1 \
    | grep -E "^  [a-z]" || echo "  FAILED: $SCEN $FAM"
done <<< "$CELLS"
echo "== wall: $(( $(date +%s) - START ))s =="

echo "== grids =="
for PAIR in $(find "$REL/clips" -mindepth 3 -maxdepth 3 -type d | sort); do
  for FAM in $(ls "$PAIR" | sed -n 's/^invalid_\(.*\)_\(easy\|medium\|hard\)$/\1/p' | sort -u); do
    conda run -n physviol python -m physviol.cli grid "$PAIR" --family "$FAM" 2>/dev/null \
      | grep -oE '"path": "[^"]*"' | sed 's/"path": /  /'
  done
done

conda run -n physviol python -m physviol.cli validate "$REL"
