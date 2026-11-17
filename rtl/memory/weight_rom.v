// =============================================================================
// weight_rom.v -- shared weight ROM: WORDS x (LANES*W_BITS), registered read.
//
// One ROM serves every layer; the engines address it with `wrom_base + offset`
// where the layout depends on the layer type (docs/memory-layout.md sec. 3.2):
//
//   std : base + ic*9 + tap     lane = output channel
//   dw  : base + g*9  + tap     lane = channel inside group g
//   pw  : base + g*cin + ic     lane = output channel
//   fc  : base + ic             lane = class
//
// 552 of the 1024 words are used by this network (256 bit each, ~17 KB).
// =============================================================================
`timescale 1ns / 1ps

module weight_rom #(
    parameter integer WORDS    = 1024,
    parameter integer LANES    = 32,
    parameter integer W_BITS   = 8,
    parameter integer ADDR_W   = 10,
    parameter         INIT_FILE = ""
) (
    input  wire                       clk,
    input  wire                       en,
    input  wire [ADDR_W-1:0]          addr,
    output reg  [LANES*W_BITS-1:0]    rdata
);

    // block RAM, not registers: without this Vivado falls back to flops when it
    // sees the initial block below and the area explodes (33k FF for the 4 KB
    // input memory alone)
    (* ram_style = "block" *) reg [LANES*W_BITS-1:0] mem [0:WORDS-1];

    always @(posedge clk) begin
        if (en) rdata <= mem[addr];
    end

    initial begin
        if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
    end

endmodule
