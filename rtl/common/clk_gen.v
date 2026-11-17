// =============================================================================
// clk_gen.v -- 200 MHz board clock (differential) to the accelerator's 48 MHz,
//              plus a reset synchroniser.
//
// The PA-StarLite board feeds a 200 MHz differential clock on R4/T4 with
// DIFF_SSTL15 levels.  The MMCM settings below give an exact 48.000 MHz:
//
//     VCO = 200 MHz * 3.0 / 1 = 600 MHz          (600..1200 MHz is legal)
//     CLKOUT0 = 600 MHz / 12.5 = 48.000 MHz      (12.5 is allowed: 0.125 steps)
//
// A lower VCO is chosen on purpose: 600 MHz sits at the bottom of the MMCM range,
// which leaves power headroom, and 12.5 is exactly representable so the output
// frequency has zero error.  (200*6/25 = 48 would also work but needs a 1200 MHz
// VCO.)
//
// Simulation: iverilog and Verilator have no UNISIM primitives, so compiling with
// `-DSIM` replaces the whole block with a straight-through clock.  The testbenches
// that care about the accelerator drive `clk` themselves and never instantiate
// this module; the SIM branch only exists so board_top.v compiles and elaborates.
// =============================================================================
`timescale 1ns / 1ps

module clk_gen (
    input  wire clk_p,          // 200 MHz differential, pin R4
    input  wire clk_n,          //                             T4
    input  wire rst_async_n,    // board reset key, pin R14
    output wire clk,            // 48 MHz
    output wire rst_n,          // synchronous, released after the MMCM locks
    output wire locked
);

`ifdef SIM
    // behavioural stand-in: the testbench is expected to drive clk_p at 48 MHz
    assign clk    = clk_p;
    assign locked = 1'b1;
    reg [2:0] rs;
    always @(posedge clk or negedge rst_async_n)
        if (!rst_async_n) rs <= 3'b000;
        else              rs <= {rs[1:0], 1'b1};
    assign rst_n = rs[2];
`else
    wire clk200;
    IBUFDS #(
        .DIFF_TERM   ("FALSE"),
        .IBUF_LOW_PWR("TRUE")
    ) u_ibufds (
        .O  (clk200),
        .I  (clk_p),
        .IB (clk_n)
    );

    wire clkfb, locked_int, clk_raw;

    MMCME2_BASE #(
        .BANDWIDTH         ("OPTIMIZED"),
        .CLKFBOUT_MULT_F   (3.000),
        .CLKFBOUT_PHASE    (0.0),
        .CLKIN1_PERIOD     (5.000),          // 200 MHz -> 5 ns
        .CLKOUT0_DIVIDE_F  (12.500),         // 600 / 12.5 = 48.000 MHz
        .CLKOUT0_DUTY_CYCLE(0.500),
        .CLKOUT0_PHASE     (0.0),
        .DIVCLK_DIVIDE     (1),
        .REF_JITTER1       (0.010),
        .STARTUP_WAIT      ("FALSE")
    ) u_mmcm (
        .CLKOUT0  (clk_raw),
        .CLKFBOUT (clkfb),
        .CLKFBIN  (clkfb),
        .CLKIN1   (clk200),
        .LOCKED   (locked_int),
        .PWRDWN   (1'b0),
        .RST      (1'b0)
    );

    // MMCME2_BASE does not contain an output buffer: the global clock buffer has
    // to be instantiated explicitly, otherwise the 48 MHz net is not on a clock
    // route and Vivado either warns or inserts one behind your back.
    BUFG u_bufg (.I (clk_raw), .O (clk));

    assign locked = locked_int;

    // hold reset until the MMCM has locked, then release synchronously
    reg [2:0] rs;
    always @(posedge clk or negedge rst_async_n)
        if (!rst_async_n) rs <= 3'b000;
        else if (!locked_int) rs <= 3'b000;
        else              rs <= {rs[1:0], 1'b1};

    assign rst_n = rs[2];
`endif

endmodule
