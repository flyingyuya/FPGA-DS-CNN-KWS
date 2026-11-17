// =============================================================================
// pe_array.v -- LANES parallel MAC lanes: 32 x (INT8 x UINT8 -> 16 bit, two
//               24-bit accumulators, variable shift + ReLU).
//
// One array serves every layer of the network.  What a lane *means* depends on
// the dataflow the engine drives (see docs/memory-layout.md sec. 4):
//
//   std (stem) : lane = output channel, one activation broadcast to all lanes
//   dw         : lane = channel, a 256-bit activation word feeds all lanes
//   pw         : lane = output channel, one activation broadcast to all lanes
//   fc         : lane = class (12 lanes used)
//
// The array is deliberately dumb: it knows nothing about pixels, taps or layer
// geometry.  It receives one weight word, one activation word and the job
// control for the current cycle, and returns one 32-channel output word when
// the engine drains a bank.  All addressing lives in the engines.
//
// Constant loading: each lane needs its own 24-bit bias and 4-bit shift, i.e.
// 896 flops for a 32-lane group.  They are written ONE LANE PER CYCLE from the
// bias ROM (`load_const` + `load_lane`), so no 896-bit staging register is
// needed: 32 cycles per (layer, channel-group), a few hundred cycles per
// inference.
//
// ONE copy is enough, and here is why.  Within a channel group every job uses
// the same constants, so a job that is still being drained and the next job
// that is already accumulating agree -- the drain/accumulate overlap keeps
// working.  Constants only change at a group boundary, and there the engine
// drains first and loads afterwards (33 cycles once per group).  Duplicating
// the registers per bank would only save those 33 cycles while costing 896
// flops, so it is not worth it.
// =============================================================================
`timescale 1ns / 1ps

module pe_array #(
    parameter integer LANES      = 32,
    parameter integer W_BITS     = 8,
    parameter integer ACT_BITS   = 8,
    parameter integer ACC_BITS   = 24,
    parameter integer SHIFT_BITS = 4
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- per-lane constants -------------------------------------------------
    input  wire                          load_const,  // write ONE lane this cycle
    input  wire [4:0]                    load_lane,   // which lane
    input  wire signed [ACC_BITS-1:0]    load_bias,
    input  wire [SHIFT_BITS-1:0]         load_shift,

    // ---- job control -------------------------------------------------------
    input  wire                          en,
    input  wire                          init,
    input  wire                          sel,         // accumulate bank
    input  wire                          rd_sel,      // read-out bank

    // ---- data --------------------------------------------------------------
    input  wire [LANES*W_BITS-1:0]       w_word,
    input  wire [LANES*ACT_BITS-1:0]     a_word,

    // ---- results -----------------------------------------------------------
    output wire [LANES*ACT_BITS-1:0]     y_word,      // lane i: [i*ACT_BITS +: ACT_BITS]
    output wire [LANES*16-1:0]           ys_word,     // unclamped, for the classifier
    output wire [LANES*ACC_BITS-1:0]     acc_a_word,
    output wire [LANES*ACC_BITS-1:0]     acc_b_word,
    output wire                          ovf
);

    wire [LANES-1:0] ovf_lane;

    reg signed [ACC_BITS-1:0] bias_q [0:LANES-1];
    reg        [SHIFT_BITS-1:0] shift_q [0:LANES-1];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < LANES; i = i + 1) begin
                bias_q[i]  <= {ACC_BITS{1'b0}};
                shift_q[i] <= {SHIFT_BITS{1'b0}};
            end
        end else if (load_const) begin
            bias_q[load_lane]  <= load_bias;
            shift_q[load_lane] <= load_shift;
        end
    end

    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : g_lane
            mac_lane #(
                .W_BITS (W_BITS), .ACT_BITS (ACT_BITS),
                .ACC_BITS (ACC_BITS), .SHIFT_BITS (SHIFT_BITS)
            ) u_lane (
                .clk        (clk),
                .rst_n      (rst_n),
                .en         (en),
                .init       (init),
                .sel        (sel),
                .w_i        (w_word[l*W_BITS +: W_BITS]),
                .a_i        (a_word[l*ACT_BITS +: ACT_BITS]),
                .bias_i     (bias_q[l]),
                .shift_i    (shift_q[l]),
                .rd_sel     (rd_sel),
                .y_o        (y_word[l*ACT_BITS +: ACT_BITS]),
                .ys_o       (ys_word[l*16 +: 16]),
                .acc_a_o    (acc_a_word[l*ACC_BITS +: ACC_BITS]),
                .acc_b_o    (acc_b_word[l*ACC_BITS +: ACC_BITS]),
                .ovf_o      (ovf_lane[l])
            );
        end
    endgenerate

    assign ovf = |ovf_lane;

endmodule
