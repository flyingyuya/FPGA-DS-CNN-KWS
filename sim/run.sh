#!/usr/bin/env bash
# Run the RTL testbenches.  Always run from the project root:
#     bash sim/run.sh            # all testbenches
#     bash sim/run.sh pe         # just the PE array
#     VCD=1 bash sim/run.sh pe   # also dump sim/build/tb_pe_array.vcd
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mkdir -p sim/build

# Verilog-2001, same dialect the RTL must stay compatible with (see
# .rules.verible_lint).  -I rtl lets the sources `include "kws_params.vh"`.
iverilog_flags=(-g2005 -Wall -Wno-timescale -I rtl -o sim/build/tb.vvp)

pe_src=(rtl/cnn/mac_lane.v rtl/cnn/pe_array.v sim/tb/tb_pe_array.v)
conv_src=(rtl/cnn/mac_lane.v rtl/cnn/pe_array.v rtl/cnn/conv_engine.v
          rtl/memory/act_bank.v rtl/memory/weight_rom.v rtl/memory/const_loader.v
          sim/tb/tb_conv_engine.v)
top_src=(rtl/cnn/accelerator_top.v rtl/cnn/layer_cfg.v rtl/cnn/mac_lane.v
         rtl/cnn/pe_array.v rtl/cnn/conv_engine.v rtl/cnn/argmax_tree.v
         rtl/memory/act_bank.v rtl/memory/weight_rom.v rtl/memory/const_loader.v
         rtl/memory/fmem.v sim/tb/tb_accelerator_top.v)

run_pe() {
    echo "=== tb_pe_array ==="
    iverilog "${iverilog_flags[@]}" "${pe_src[@]}"
    if [ "${VCD:-0}" = "1" ]; then
        vvp sim/build/tb.vvp +vcd
    else
        vvp sim/build/tb.vvp
    fi
}

run_conv() {
    echo "=== tb_conv_engine ==="
    iverilog "${iverilog_flags[@]}" "${conv_src[@]}"
    if [ "${VCD:-0}" = "1" ]; then
        vvp sim/build/tb.vvp +vcd
    else
        vvp sim/build/tb.vvp
    fi
}

run_top() {
    echo "=== tb_accelerator_top ==="
    iverilog "${iverilog_flags[@]}" "${top_src[@]}"
    if [ "${VCD:-0}" = "1" ]; then
        vvp sim/build/tb.vvp +vcd
    else
        vvp sim/build/tb.vvp
    fi
}

case "${1:-all}" in
    pe)    run_pe ;;
    conv)  run_conv ;;
    top)   run_top ;;
    all)   run_pe; run_conv; run_top ;;
    *)   echo "usage: bash sim/run.sh [all|pe|conv|top]   (tb_*.v live in sim/)"; exit 1 ;;
esac
