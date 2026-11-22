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
set_property -dict {PACKAGE_PIN W21 IOSTANDARD LVCMOS33}    [get_ports key1]
# KEY2 (Y21) toggles single-shot <-> continuous recognition
set_property -dict {PACKAGE_PIN Y21 IOSTANDARD LVCMOS33}    [get_ports key2]
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
set_false_path -from [get_ports {key1 key2}]

# (there was a set_input_delay -clock clk48 here; no such clock exists -- the
#  system clock is the MMCM output -- so it only produced a critical warning
#  every run.  Both keys are already excluded from timing with set_false_path.)

# ---- no use of the transceivers, keep the report clean ----------------------
# (was: set_property SEVERITY {Info} [get_drc_checks NSTD-1] -- NSTD-1 no longer
#  exists as a check, so it produced "Invalid severity Info" on every run)

# =============================================================================
# WM8960 audio codec (JM2 connector)
# -----------------------------------------------------------------------------
# Pin numbers come from the board vendor's connector spreadsheet
# ("Puzhi PA-Starlite Connectors Pins Signal and Equal Length.xlsx", JM2 sheet),
# matching the earlier Sonic-Speed-Parallelism project's constrs/audio.xdc.
#
# J20 / K18 are clock-capable pins (SRCC / MRCC), which is why BCLK and LRCLK sit
# there: i2s_rx uses BCLK as a real clock, so it gets a create_clock below and the
# 2-FF toggle synchroniser into the 48 MHz domain becomes a false path.
# This project runs the codec at 16 kHz, so BCLK is 1.024 MHz (not 3.072 MHz).
# =============================================================================
set_property -dict {PACKAGE_PIN J20 IOSTANDARD LVCMOS33} [get_ports aud_bclk]
set_property -dict {PACKAGE_PIN K18 IOSTANDARD LVCMOS33} [get_ports aud_lrclk]
set_property -dict {PACKAGE_PIN H20 IOSTANDARD LVCMOS33} [get_ports aud_adcdat]
set_property -dict {PACKAGE_PIN K21 IOSTANDARD LVCMOS33} [get_ports aud_scl]
set_property -dict {PACKAGE_PIN K22 IOSTANDARD LVCMOS33} [get_ports aud_sda]

# 16 kHz * 64 bit/slot = 1.024 MHz BCLK; LRCLK (16 kHz) is data as far as the
# tools are concerned -- the BCLK-domain logic samples it as an ordinary signal.
create_clock -name aud_bclk -period 976.562 -waveform {0.000 488.281} [get_ports aud_bclk]
set_false_path -from [get_ports aud_lrclk] -to [get_clocks aud_bclk]

# the only crossing is the single-bit "new sample" toggle through 2 FFs
set_false_path -from [get_clocks aud_bclk] -to [get_clocks -of_objects [get_pins u_clk/u_bufg/O]]
set_false_path -from [get_clocks -of_objects [get_pins u_clk/u_bufg/O]] -to [get_clocks aud_bclk]

# open-drain I2C: the codec has its own pull-ups, no internal pull needed
set_property SLEW SLOW [get_ports {aud_scl}]

# =============================================================================
# LCD (480x272, 24-bit RGB, no DE pin)
# Pin numbers copied from Sonic-Speed-Parallelism/constrs/lcd.xdc, itself taken
# from the board vendor's 3_16_PZ_LCD example.
# =============================================================================
set_property -dict {PACKAGE_PIN F13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[0]}]
set_property -dict {PACKAGE_PIN E16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[1]}]
set_property -dict {PACKAGE_PIN F14 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[2]}]
set_property -dict {PACKAGE_PIN D16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[3]}]
set_property -dict {PACKAGE_PIN D14 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[4]}]
set_property -dict {PACKAGE_PIN C13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[5]}]
set_property -dict {PACKAGE_PIN D15 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[6]}]
set_property -dict {PACKAGE_PIN B13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[7]}]
set_property -dict {PACKAGE_PIN C14 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[8]}]
set_property -dict {PACKAGE_PIN A13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[9]}]
set_property -dict {PACKAGE_PIN C15 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[10]}]
set_property -dict {PACKAGE_PIN A14 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[11]}]
set_property -dict {PACKAGE_PIN E13 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[12]}]
set_property -dict {PACKAGE_PIN A15 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[13]}]
set_property -dict {PACKAGE_PIN E14 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[14]}]
set_property -dict {PACKAGE_PIN A16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[15]}]
set_property -dict {PACKAGE_PIN B15 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[16]}]
set_property -dict {PACKAGE_PIN B17 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[17]}]
set_property -dict {PACKAGE_PIN B16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[18]}]
set_property -dict {PACKAGE_PIN B18 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[19]}]
set_property -dict {PACKAGE_PIN F16 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[20]}]
set_property -dict {PACKAGE_PIN D17 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[21]}]
set_property -dict {PACKAGE_PIN E17 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[22]}]
set_property -dict {PACKAGE_PIN C17 IOSTANDARD LVCMOS33} [get_ports {lcd_rgb[23]}]

# LCD pixel clock: 12.5 MHz from the MMCM (600 / 48).  Separate domain, so define
# the clock and declare the crossing asynchronous -- only the ~1 s word payload
# crosses it, through a 2-FF synchroniser.
set_property -dict {PACKAGE_PIN C18 IOSTANDARD LVCMOS33} [get_ports lcd_hsync]
set_property -dict {PACKAGE_PIN E18 IOSTANDARD LVCMOS33} [get_ports lcd_vsync]
set_property -dict {PACKAGE_PIN F18 IOSTANDARD LVCMOS33} [get_ports lcd_clk]

create_clock -name lcd_pix -period 80.000 [get_ports lcd_clk]
set_clock_groups -asynchronous \
    -group [get_clocks -of_objects [get_pins u_clk/u_bufg_pix/O]] \
    -group [get_clocks -of_objects [get_pins u_clk/u_bufg/O]]

