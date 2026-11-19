// =============================================================================
// xilinx_stubs.v -- simulation-only stand-ins for the three unisim primitives
// board_top uses (IBUFDS, MMCME2_BASE, BUFG).  clk_gen.v already has a
// SIMULATION branch that bypasses them, but elaborating board_top as a whole is
// still useful as a cheap "did I wire the hierarchy right" check, and that needs
// the primitives to exist.  Behaviourally they are pass-throughs with LOCKED=1.
// =============================================================================
`timescale 1ns / 1ps

module IBUFDS #(parameter DIFF_TERM = "FALSE", parameter IBUF_LOW_PWR = "TRUE") (
    input  wire I, input wire IB, output wire O);
    assign O = I;
endmodule

module BUFG (input wire I, output wire O);
    assign O = I;
endmodule

module MMCME2_BASE #(
    parameter BANDWIDTH          = "OPTIMIZED",
    parameter real CLKFBOUT_MULT_F   = 5.0,
    parameter real CLKFBOUT_PHASE    = 0.0,
    parameter real CLKIN1_PERIOD     = 10.0,
    parameter real CLKOUT0_DIVIDE_F  = 5.0,
    parameter real CLKOUT0_DUTY_CYCLE= 0.5,
    parameter real CLKOUT0_PHASE     = 0.0,
    parameter integer DIVCLK_DIVIDE  = 1,
    parameter real REF_JITTER1       = 0.010,
    parameter STARTUP_WAIT           = "FALSE"
) (
    output wire CLKOUT0,
    output wire CLKFBOUT,
    input  wire CLKFBIN,
    input  wire CLKIN1,
    output wire LOCKED,
    input  wire PWRDWN,
    input  wire RST);
    assign CLKOUT0  = CLKIN1;
    assign CLKFBOUT = CLKIN1;
    assign LOCKED   = 1'b1;
endmodule
