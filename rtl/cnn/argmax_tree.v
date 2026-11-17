// =============================================================================
// argmax_tree.v -- 12-way maximum with margin, for the fully connected layer.
//
// The accelerator classifies by comparing the shifted logits, so the softmax of
// the reference implementation (exponential lookup table + 32-bit divider + INT16
// probability ROM) is replaced by a comparison pass.  Ties resolve to the lowest
// class index, which is what the golden vectors expect.
//
//   class  = index of the maximum                       4 bit
//   max    = that maximum                              16 bit
//   margin = max - second best, a free confidence proxy 16 bit
//
// ---------------------------------------------------------------------------
// Sequential on purpose
// ---------------------------------------------------------------------------
// The first version was one combinational priority chain.  Place & route put it
// on a 44-level path (26.7 ns) and it missed 48 MHz by 5.9 ns.  Since the
// classifier runs ONCE per inference -- 55,522 cycles -- there is no reason to
// pay for that: one comparison per cycle finishes in NUM_CLASSES cycles, costs a
// tenth of the logic (one comparator instead of twelve), and every path is short.
//
//   start ----> i = 1..NUM_CLASSES-1, one step per cycle ----> done
// =============================================================================
`timescale 1ns / 1ps

module argmax_tree #(
    parameter integer NUM_CLASSES = 12,
    parameter integer LOGIT_BITS  = 16,
    parameter integer CLS_BITS    = 4
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,        // pulse, logits must be stable
    input  wire [NUM_CLASSES*LOGIT_BITS-1:0] logits_i,
    output reg  [CLS_BITS-1:0]               class_o,
    output reg  signed [LOGIT_BITS-1:0]      max_o,
    output reg  signed [LOGIT_BITS-1:0]      margin_o,
    output reg                               done
);

    localparam signed [LOGIT_BITS-1:0] MOST_NEG = {1'b1, {(LOGIT_BITS-1){1'b0}}};

    reg [7:0] i;
    reg       busy;
    reg       fin;          // one extra cycle so the result uses the updated regs
    reg signed [LOGIT_BITS-1:0] best, second;
    reg [CLS_BITS-1:0]          bidx;

    wire signed [LOGIT_BITS-1:0] v = $signed(logits_i[i[$clog2(NUM_CLASSES)-1:0]*LOGIT_BITS
                                                      +: LOGIT_BITS]);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i        <= 8'd1;
            busy     <= 1'b0;
            fin      <= 1'b0;
            done     <= 1'b0;
            best     <= {LOGIT_BITS{1'b0}};
            second   <= MOST_NEG;
            bidx     <= {CLS_BITS{1'b0}};
            class_o  <= {CLS_BITS{1'b0}};
            max_o    <= {LOGIT_BITS{1'b0}};
            margin_o <= {LOGIT_BITS{1'b0}};
        end else begin
            done <= 1'b0;
            if (fin) begin
                // the scan just finished; publish the result from the settled regs
                fin      <= 1'b0;
                done     <= 1'b1;
                class_o  <= bidx;
                max_o    <= best;
                margin_o <= best - second;
            end else if (!busy) begin
                if (start) begin
                    busy   <= 1'b1;
                    i      <= 8'd1;
                    best   <= $signed(logits_i[0 +: LOGIT_BITS]);
                    bidx   <= {CLS_BITS{1'b0}};
                    second <= MOST_NEG;
                end
            end else begin
                // one comparison per cycle over classes 1 .. NUM_CLASSES-1
                if (v > best) begin
                    second <= best;
                    best   <= v;
                    bidx   <= i[CLS_BITS-1:0];
                end else if (v > second) begin
                    second <= v;
                end

                if (i == NUM_CLASSES - 1) begin
                    busy <= 1'b0;
                    fin  <= 1'b1;
                end else begin
                    i <= i + 8'd1;
                end
            end
        end
    end

endmodule
