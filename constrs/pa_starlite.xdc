# =============================================================================
# pa_starlite.xdc -- 璞致 PA-StarLite, XC7A100T-2FGG484
#
# Pin assignments and IO standards are copied from the board vendor's own example
# projects (1_FPGA/3_1_PZ_LED, 3_3_PZ_KEY, 3_10_PZ_UART), so they match the
# hardware rather than a datasheet guess.
#
#   200 MHz differential clock   R4 (P) / T4 (N)   DIFF_SSTL15
#   reset key                    R14               LVCMOS33
#   key[0]                       W21               LVCMOS33
#   led[0] / led[1]              W22 / Y22         LVCMOS33
#   uart tx / rx                 P15 / P14         LVCMOS33
#
# Clocking: the board clock is 200 MHz; clk_gen's MMCM divides it to exactly
# 48.000 MHz.  The MMCM output is declared as a generated clock so that the
# timing analysis in report/ is meaningful.
# =============================================================================

# ---- configuration of the QSPI flash (board default: x4, 3.3 V) -------------
set_property CFGBVS VCCO            [current_design]
set_property CONFIG_VOLTAGE 3.3     [current_design]
set_property BITSTREAM.GENERAL.COMPRESS         true [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE        50   [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH      4    [current_design]
set_property BITSTREAM.CONFIG.SPI_FALL_EDGE     Yes  [current_design]

# ---- pins -------------------------------------------------------------------
set_property -dict {PACKAGE_PIN R4  IOSTANDARD DIFF_SSTL15} [get_ports sys_clk_p]
set_property -dict {PACKAGE_PIN T4  IOSTANDARD DIFF_SSTL15} [get_ports sys_clk_n]
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33}    [get_ports sys_rstn]
set_property -dict {PACKAGE_PIN W21 IOSTANDARD LVCMOS33}    [get_ports key]
set_property -dict {PACKAGE_PIN P15 IOSTANDARD LVCMOS33}    [get_ports uart_tx]
set_property -dict {PACKAGE_PIN W22 IOSTANDARD LVCMOS33}    [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN Y22 IOSTANDARD LVCMOS33}    [get_ports {led[1]}]

# ---- clocks -----------------------------------------------------------------
# 200 MHz board oscillator
create_clock -name sys_clk -period 5.000 [get_ports sys_clk_p]

# The MMCM output clock (48.000 MHz) needs no explicit constraint: Vivado derives
# it from the MMCME2_BASE attributes (CLKFBOUT_MULT_F=3.0, CLKOUT0_DIVIDE_F=12.5)
# and from the create_clock above.  Adding a manual create_generated_clock on the
# same net would create a duplicate clock and confuse the timing report.

# the reset key is asynchronous to everything
set_false_path -from [get_ports sys_rstn]
set_false_path -from [get_ports key]

# inputs that are sampled by clk48 only
set_input_delay -clock clk48 2.000 [get_ports key]

# ---- no use of the transceivers, keep the report clean ----------------------
set_property SEVERITY {Info} [get_drc_checks NSTD-1]
