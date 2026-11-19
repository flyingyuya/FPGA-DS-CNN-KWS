#!/usr/bin/env bash
# prog.sh -- convenience wrapper around program.tcl
#
#   bash scripts/prog.sh            # program (volatile)
#   bash scripts/prog.sh detect     # only probe the JTAG chain
#   bash scripts/prog.sh flash      # write QSPI (survives power-off)
#   bash scripts/prog.sh <file>.bit # program a specific bitstream
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIVADO="${VIVADO:-vivado}"
if ! command -v "$VIVADO" >/dev/null; then
    echo "❌ vivado not on PATH; set VIVADO=/path/to/vivado" >&2
    exit 1
fi
cd "$ROOT"
"$VIVADO" -mode batch -notrace -source scripts/program.tcl -tclargs "$@" 2>&1 \
  | grep -vE "^(INFO: \[Labtools|WARNING: \[Labtools)" || true
