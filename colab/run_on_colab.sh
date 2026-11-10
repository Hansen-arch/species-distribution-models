#!/usr/bin/env bash
# Run the species distribution panel on a Google Colab VM and pull the results back.
#
#   bash colab/run_on_colab.sh
#   SDM_ARGS="--species welwitschia-mirabilis --fast" bash colab/run_on_colab.sh
#
# Requires an authenticated colab CLI. CPU sessions do not consume compute units,
# so this works even when the account balance reads 0.00.
set -euo pipefail

COLAB="${COLAB:-$HOME/Documents/mcp-servers/colab-cli-mcp/.venv/bin/colab}"
SESSION="${SESSION:-sdm}"
SDM_ARGS="${SDM_ARGS:---species all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="/tmp/sdm_bundle.tar.gz"

echo "bundle from $ROOT"
tar -czf "$BUNDLE" -C "$ROOT" \
  --exclude=data --exclude=results --exclude=.git \
  --exclude=__pycache__ .

"$COLAB" new -s "$SESSION"
"$COLAB" upload -s "$SESSION" "$BUNDLE" /content/sdm_bundle.tar.gz
"$COLAB" exec -s "$SESSION" --timeout 5400 -f "$ROOT/colab/remote_run.py" --env "SDM_ARGS=$SDM_ARGS"
"$COLAB" download -s "$SESSION" /content/sdm_results.tar.gz "$ROOT/results_colab.tar.gz"
"$COLAB" stop -s "$SESSION"

echo "unpack results"
mkdir -p "$ROOT/results_colab"
tar -xzf "$ROOT/results_colab.tar.gz" -C "$ROOT/results_colab"
echo "done, see results_colab/results"
