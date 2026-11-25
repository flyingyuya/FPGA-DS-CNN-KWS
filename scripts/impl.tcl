# =============================================================================
# impl.tcl -- synthesis + place & route for 璞致 PA-StarLite (XC7A100T-2FGG484)
#
#   vivado -mode batch -notrace -source scripts/impl.tcl
#
# Run from the project root: the memory images are referenced relatively
# ("mem/weights/w_all.mem") and are read at synthesis time.
# =============================================================================
set part   xc7a100tfgg484-2
set outdir report
file mkdir $outdir

set_property include_dirs [list rtl rtl/wm8960 rtl/preprocessing] [current_fileset]

read_verilog [list \
    rtl/board_top.v \
    rtl/common/clk_gen.v \
    rtl/cnn/accelerator_top.v \
    rtl/cnn/layer_cfg.v \
    rtl/cnn/mac_lane.v \
    rtl/cnn/pe_array.v \
    rtl/cnn/conv_engine.v \
    rtl/cnn/argmax_tree.v \
    rtl/common/uart_tx.v \
    rtl/memory/act_bank.v \
    rtl/memory/weight_rom.v \
    rtl/memory/const_loader.v \
    rtl/memory/fmem.v \
    rtl/common/i2s_rx.v \
    rtl/common/key_debounce.v \
    rtl/common/audio_fifo.v \
    rtl/wm8960/i2c_bit_shift.v \
    rtl/wm8960/i2c_control.v \
    rtl/wm8960/WM8960_init_table.v \
    rtl/wm8960/WM8960_init.v \
    rtl/preprocessing/fft_core.v \
    rtl/preprocessing/fft_addr_gen.v \
    rtl/preprocessing/fft_butterfly.v \
    rtl/preprocessing/fft_twiddle_rom.v \
    rtl/preprocessing/mel_fb.v \
    rtl/preprocessing/mel_rom.v \
    rtl/preprocessing/log_q8.v \
    rtl/preprocessing/log_lut.v \
    rtl/preprocessing/dct.v \
    rtl/preprocessing/mfcc_roms.v \
    rtl/preprocessing/mfcc_top.v \
    rtl/lcd/lcd_timing.v rtl/lcd/font_rom.v \
    rtl/lcd/text_overlay.v rtl/lcd/word_to_text.v \
]
read_xdc constrs/pa_starlite.xdc

synth_design -top board_top -part $part
write_checkpoint -force $outdir/post_synth.dcp
report_utilization    -file $outdir/synth_util.rpt
report_timing_summary -file $outdir/synth_timing.rpt
report_clocks         -file $outdir/synth_clocks.rpt

opt_design
place_design
phys_opt_design
route_design

write_checkpoint     -force $outdir/post_route.dcp
report_utilization    -file $outdir/impl_util.rpt
report_timing_summary -file $outdir/impl_timing.rpt -max_paths 10 -warn_on_violation
report_drc            -file $outdir/impl_drc.rpt
report_power          -file $outdir/impl_power.rpt
write_bitstream -force $outdir/board_top.bit

puts "=== IMPL DONE ==="
