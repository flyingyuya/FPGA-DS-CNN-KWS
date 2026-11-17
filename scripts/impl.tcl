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

set_property include_dirs [list rtl] [current_fileset]

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
