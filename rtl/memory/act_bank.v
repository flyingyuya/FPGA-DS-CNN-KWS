// =============================================================================
// act_bank.v -- one activation buffer: WORDS x (LANES*ACT_BITS) with one write
//               port (byte enables) and one read port, both registered.
//
// Layout (docs/memory-layout.md sec. 3.1): word = 32 channels x 8 bit, the byte
// at [l*8 +: 8] is channel l of that pixel group.  Everything downstream
// assumes `word_addr = pixel*NG + group`, but that is the engines' business --
// this module only moves words.
//
// Byte enables exist because a layer whose channel count is not a multiple of 32
// only fills part of its last word; the engine masks the rest so the bank image
// stays deterministic (the golden vectors depend on it).
//
// 1024 x 256 bit = 256 Kb = 8 BRAM36.  Vivado infers block RAM from the
// registered read; the write port maps to the byte-write-enable structure.
// =============================================================================
`timescale 1ns / 1ps

module act_bank #(
    parameter integer WORDS     = 1024,
    parameter integer LANES     = 32,
    parameter integer ACT_BITS  = 8,
    parameter integer ADDR_W    = 10,
    parameter         INIT_FILE = ""        // simulation only
) (
    input  wire                          clk,

    input  wire                          we,
    input  wire [LANES-1:0]              wbe,
    input  wire [ADDR_W-1:0]             waddr,
    input  wire [LANES*ACT_BITS-1:0]     wdata,

    input  wire                          re,
    input  wire [ADDR_W-1:0]             raddr,
    output reg  [LANES*ACT_BITS-1:0]     rdata
);

    // block RAM, not registers: without this Vivado falls back to flops when it
    // sees the initial block below and the area explodes (33k FF for the 4 KB
    // input memory alone)
    (* ram_style = "block" *) reg [LANES*ACT_BITS-1:0] mem [0:WORDS-1];

    integer l;
    always @(posedge clk) begin
        if (we) begin
            for (l = 0; l < LANES; l = l + 1)
                if (wbe[l]) mem[waddr][l*ACT_BITS +: ACT_BITS] <= wdata[l*ACT_BITS +: ACT_BITS];
        end
        if (re) rdata <= mem[raddr];
    end

    // Zero first, then optionally load an image.  This matters beyond simulation:
    // the engine redirects out-of-range taps to the reserved last word, so that
    // word must read as zero.  Block RAM comes up as zero on an FPGA anyway
    // (Vivado only emits INIT_xx attributes for locations that differ), so this
    // initial block describes the hardware instead of inventing a simulation-only
    // behaviour.
    integer k;
    initial begin
        for (k = 0; k < WORDS; k = k + 1) mem[k] = {LANES*ACT_BITS{1'b0}};
        if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
    end

endmodule
