# =============================================================================
# util_hier.tcl -- per-module resource report, synthesis only (~2 minutes).
#
#   vivado -mode batch -notrace -source scripts/util_hier.tcl
#   -> report/util_hier.rpt
#
# Why this exists: memory-inference failures are SILENT.  Two real examples from
# this project:
#   * a 256x48 debug array became 12,288 flip-flops instead of a BRAM,
#   * fmem's write port written as if/else-if became "two write ports", which a
#     simple dual-port BRAM cannot map, so it fell back to LUTRAM (1,536 LUTs).
# Neither produced an error or even a warning.  Looking at the `LUTRAMs` column
# per hierarchy is the only way to catch them, and it is much cheaper than the
# 10-minute place & route just to read impl_util.rpt.
# =============================================================================
set part   xc7a100tfgg484-2
set outdir report
file mkdir $outdir

set_property include_dirs [list rtl rtl/wm8960 rtl/preprocessing] [current_fileset]

read_verilog [list \
    rtl/board_top.v \
    rtl/common/clk_gen.v rtl/common/uart_tx.v rtl/common/i2s_rx.v \
    rtl/common/audio_fifo.v rtl/common/key_debounce.v \
    rtl/wm8960/i2c_bit_shift.v rtl/wm8960/i2c_control.v \
    rtl/wm8960/WM8960_init_table.v rtl/wm8960/WM8960_init.v \
    rtl/cnn/accelerator_top.v rtl/cnn/layer_cfg.v rtl/cnn/mac_lane.v \
    rtl/cnn/pe_array.v rtl/cnn/conv_engine.v rtl/cnn/argmax_tree.v \
    rtl/memory/act_bank.v rtl/memory/weight_rom.v rtl/memory/const_loader.v \
    rtl/memory/fmem.v \
    rtl/preprocessing/fft_core.v rtl/preprocessing/fft_addr_gen.v \
    rtl/preprocessing/fft_butterfly.v rtl/preprocessing/fft_twiddle_rom.v \
    rtl/preprocessing/mel_fb.v rtl/preprocessing/mel_rom.v \
    rtl/preprocessing/log_q8.v rtl/preprocessing/log_lut.v \
    rtl/preprocessing/dct.v rtl/preprocessing/mfcc_roms.v \
    rtl/preprocessing/mfcc_top.v \
    rtl/lcd/lcd_timing.v rtl/lcd/font_rom.v \
    rtl/lcd/word_to_text.v rtl/lcd/text_overlay.v \
]
read_xdc constrs/pa_starlite.xdc

synth_design -top board_top -part $part
report_utilization -hierarchical -hierarchical_depth 3 -file $outdir/util_hier.rpt
report_utilization -file $outdir/util_hier_top.rpt

puts ""
puts "  -> report/util_hier.rpt      (per-instance LUT / FF / RAMB36 / LUTRAM)"
puts "  -> report/util_hier_top.rpt  (device totals)"
puts ""
puts "  Read the LUTRAMs column: a memory that failed to infer as BRAM shows up"
puts "  there (and as a large LUT count) with no error or warning."
puts "=== UTIL DONE ==="
