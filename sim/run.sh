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
iverilog_flags=(-g2005 -Wall -Wno-timescale -I rtl -I rtl/preprocessing -o sim/build/tb.vvp)

pe_src=(rtl/cnn/mac_lane.v rtl/cnn/pe_array.v sim/tb/tb_pe_array.v)
conv_src=(rtl/cnn/mac_lane.v rtl/cnn/pe_array.v rtl/cnn/conv_engine.v
          rtl/memory/act_bank.v rtl/memory/weight_rom.v rtl/memory/const_loader.v
          sim/tb/tb_conv_engine.v)
mel_src=(rtl/preprocessing/fft_core.v rtl/preprocessing/fft_addr_gen.v
         rtl/preprocessing/fft_butterfly.v rtl/preprocessing/fft_twiddle_rom.v
         rtl/preprocessing/mel_fb.v rtl/preprocessing/mel_rom.v
         rtl/common/audio_fifo.v sim/tb/tb_mel_fb.v)
fft_src=(rtl/preprocessing/fft_core.v rtl/preprocessing/fft_addr_gen.v
         rtl/preprocessing/fft_butterfly.v rtl/preprocessing/fft_twiddle_rom.v
         rtl/common/audio_fifo.v sim/tb/tb_fft_core.v)
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

run_fft() {
    echo "=== tb_fft_core ==="
    iverilog "${iverilog_flags[@]}" "${fft_src[@]}"
    vvp sim/build/tb.vvp
}

run_mel() {
    echo "=== tb_mel_fb ==="
    iverilog "${iverilog_flags[@]}" "${mel_src[@]}"
    vvp sim/build/tb.vvp
}

run_log() {
    echo "=== tb_log_q8 ==="
    iverilog "${iverilog_flags[@]}" rtl/preprocessing/log_q8.v rtl/preprocessing/log_lut.v sim/tb/tb_log_q8.v
    vvp sim/build/tb.vvp
}

run_dct() {
    echo "=== tb_dct40 ==="
    iverilog "${iverilog_flags[@]}" rtl/preprocessing/dct40.v sim/tb/tb_dct40.v
    vvp sim/build/tb.vvp
}

wm_src=(rtl/wm8960/i2c_bit_shift.v rtl/wm8960/i2c_control.v
        rtl/wm8960/WM8960_init_table.v rtl/wm8960/WM8960_init.v
        sim/tb/tb_wm8960_init.v)

run_wm() {
    echo "=== tb_wm8960_init ==="
    iverilog "${iverilog_flags[@]}" -I rtl/wm8960 "${wm_src[@]}"
    vvp sim/build/tb.vvp
}

board_src=(-DSIM -I rtl -I rtl/cnn -I rtl/memory -I rtl/common
           -I rtl/preprocessing -I rtl/wm8960
           rtl/board_top.v rtl/common/clk_gen.v rtl/common/uart_tx.v
           rtl/common/i2s_rx.v rtl/common/audio_fifo.v rtl/common/key_debounce.v
           rtl/wm8960/i2c_bit_shift.v rtl/wm8960/i2c_control.v
           rtl/wm8960/WM8960_init_table.v rtl/wm8960/WM8960_init.v
           rtl/preprocessing/fft_core.v rtl/preprocessing/fft_addr_gen.v
           rtl/preprocessing/fft_butterfly.v rtl/preprocessing/fft_twiddle_rom.v
           rtl/preprocessing/mel_fb.v rtl/preprocessing/mel_rom.v
           rtl/preprocessing/log_q8.v rtl/preprocessing/log_lut.v
           rtl/preprocessing/dct40.v rtl/preprocessing/mfcc_roms.v
           rtl/preprocessing/mfcc_top.v
           rtl/cnn/accelerator_top.v rtl/cnn/layer_cfg.v rtl/cnn/mac_lane.v
           rtl/cnn/pe_array.v rtl/cnn/conv_engine.v rtl/cnn/argmax_tree.v
           rtl/memory/act_bank.v rtl/memory/weight_rom.v
           rtl/memory/const_loader.v rtl/memory/fmem.v
           rtl/lcd/lcd_timing.v rtl/lcd/font_rom.v rtl/lcd/text_overlay.v rtl/lcd/word_to_text.v
           sim/tb/tb_board_top.v)

run_board() {
    echo "=== tb_board_top ==="
    iverilog "${iverilog_flags[@]}" "${board_src[@]}"
    vvp sim/build/tb.vvp
}

i2s_src=(rtl/common/i2s_rx.v sim/tb/tb_i2s_rx.v)

run_i2s() {
    echo "=== tb_i2s_rx ==="
    iverilog "${iverilog_flags[@]}" "${i2s_src[@]}"
    vvp sim/build/tb.vvp
}

mfcc_src=(rtl/preprocessing/mfcc_top.v rtl/preprocessing/mfcc_roms.v
          rtl/preprocessing/fft_core.v rtl/preprocessing/fft_addr_gen.v
          rtl/preprocessing/fft_butterfly.v rtl/preprocessing/fft_twiddle_rom.v
          rtl/preprocessing/mel_fb.v rtl/preprocessing/mel_rom.v
          rtl/preprocessing/log_q8.v rtl/preprocessing/log_lut.v
          rtl/preprocessing/dct40.v rtl/common/audio_fifo.v
          sim/tb/tb_mfcc_top.v)

run_mfcc() {
    echo "=== tb_mfcc_top ==="
    iverilog "${iverilog_flags[@]}" "${mfcc_src[@]}"
    vvp sim/build/tb.vvp
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
    fft)   run_fft ;;
    mel)   run_mel ;;
    log)   run_log ;;
    dct)   run_dct ;;
    wm)    run_wm ;;
    board) run_board ;;
    word) iverilog -g2005 -o /tmp/tb_word_to_text.vvp sim/tb/tb_word_to_text.v rtl/lcd/word_to_text.v
        vvp /tmp/tb_word_to_text.vvp | grep -E "PASS|FAIL" ;;

  i2s)   run_i2s ;;
    mfcc)  run_mfcc ;;
    all)   run_pe; run_conv; run_top; run_fft; run_mel; run_log; run_dct; run_i2s; run_wm; run_mfcc ;;
    *)   echo "usage: bash sim/run.sh [all|pe|conv|top|fft|mel|log|dct|word|i2s|mfcc]   (tb_*.v live in sim/)"; exit 1 ;;
esac
