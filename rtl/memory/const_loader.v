// =============================================================================
// const_loader.v -- per-channel constant ROM (bias + shift) and the shifter that
//                   pushes it into the PE array one lane per cycle.
//
// ROM word (docs/memory-layout.md sec. 3.3):
//
//     32 bit = { 4'b0, shift[3:0], bias[23:0] }
//
// bias[23:0] is the accumulator-domain constant with the input zero point and
// the 2**(shift-1) rounding term already folded in, so the PE's `>>>` is the
// whole requantisation.
//
// Loading: LANES+1 cycles per (layer, channel-group).  A 32-lane group needs 896
// flops in the PE array, and streaming one lane per cycle avoids a second
// 896-bit staging register.  The engines call this once per channel group, so
// the whole network pays only a few hundred cycles.
//
//   cycle 0        : issue address ch_base, nothing on the bus
//   cycle 1..LANES : rdata[k-1] is presented as lane k-1
// =============================================================================
`timescale 1ns / 1ps

module const_loader #(
    parameter integer DEPTH      = 512,
    parameter integer ADDR_W     = 9,
    parameter integer ACC_BITS   = 24,
    parameter integer SHIFT_BITS = 4,
    parameter integer LANES      = 32,
    parameter         INIT_FILE  = ""
) (
    input  wire                        clk,
    input  wire                        rst_n,

    input  wire                        start,        // pulse: load LANES lanes
    input  wire [ADDR_W-1:0]           ch_base,      // first channel of the group
    output reg                         busy,
    output reg                         done,

    // to the PE array's constant port
    output reg                         load_const,
    output reg  [4:0]                  load_lane,
    output reg  signed [ACC_BITS-1:0]  load_bias,
    output reg  [SHIFT_BITS-1:0]       load_shift
);

    // block RAM, not registers: without this Vivado falls back to flops when it
    // sees the initial block below and the area explodes (33k FF for the 4 KB
    // input memory alone)
    (* ram_style = "block" *) reg [31:0]       rom [0:DEPTH-1];
    reg [ADDR_W-1:0] raddr;
    reg              re;
    reg [31:0]       rdata;

    always @(posedge clk) begin
        if (re) rdata <= rom[raddr];
    end

    reg [5:0] cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy       <= 1'b0;
            done       <= 1'b0;
            cnt        <= 6'd0;
            re         <= 1'b0;
            raddr      <= {ADDR_W{1'b0}};
            load_const <= 1'b0;
            load_lane  <= 5'd0;
            load_bias  <= {ACC_BITS{1'b0}};
            load_shift <= {SHIFT_BITS{1'b0}};
        end else begin
            done <= 1'b0;
            if (!busy) begin
                load_const <= 1'b0;
                re         <= 1'b0;
                if (start) begin
                    busy  <= 1'b1;
                    cnt   <= 6'd0;
                    re    <= 1'b1;
                    raddr <= ch_base;
                end
            end else begin
                cnt   <= cnt + 6'd1;
                raddr <= raddr + 1'b1;          // always prefetch the next word
                re    <= 1'b1;
                if (cnt == 6'd0) begin
                    load_const <= 1'b0;         // first word is still in flight
                end else begin
                    load_const <= 1'b1;
                    load_lane  <= cnt[4:0] - 5'd1;
                    load_bias  <= rdata[ACC_BITS-1:0];
                    load_shift <= rdata[ACC_BITS+SHIFT_BITS-1:ACC_BITS];
                    if (cnt == LANES[5:0]) begin
                        // this very cycle is the one that loads the last lane,
                        // so busy/done retire here but load_const must stay high
                        // (the idle branch clears it on the next cycle)
                        busy <= 1'b0;
                        done <= 1'b1;
                        re   <= 1'b0;
                    end
                end
            end
        end
    end

    initial begin
        if (INIT_FILE != "") $readmemh(INIT_FILE, rom);
    end

endmodule
