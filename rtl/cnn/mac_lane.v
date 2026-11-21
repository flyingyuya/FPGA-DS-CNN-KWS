// =============================================================================
// mac_lane.v -- one INT8 x UINT8 multiply-accumulate lane with two 24-bit
//               accumulators and a shift + ReLU output stage.
//
// This is one lane of the array, with three changes that the numeric 
// contract in docs/quantisation-math.md makes possible:
//
//   * the zero point is folded into bias_i, so there is NO subtractor: the
//     multiplier is a plain 8x8 and the product fits in 16 bits
//       |w*a| <= 127*255 = 32385 < 2^15
//   * bias_i also carries 2**(shift-1), so the arithmetic right shift below is
//     already round-half-up -- no rounding adder
//   * there are TWO accumulators and the read-out happens on the *other* bank,
//     so a finished job can be drained while the next job accumulates.  That
//     removes the drain bubble: steady state is one accumulate per cycle.
//
// Timing (one accumulate per cycle, no MAC pipeline -- at 48 MHz the critical
// path is one 8x8 multiply plus a 24-bit add, roughly 4 ns of 20.8 ns):
//
//   cycle t    : w_i / a_i / en / init / sel valid
//   cycle t+1  : acc[sel] updated; the *result* of a finished job is visible
//                through y_o for as long as rd_sel points at its bank
//
// `init` marks the first tap of a job: the accumulator is seeded with bias_i
// instead of its own value, and THAT tap's product is still added on top:
//
//     acc <= (init ? bias_i : acc) + w*a
//
// so the job needs no clearing cycle and no separate bias adder -- only a
// 24-bit 2:1 mux on the accumulate operand.  (Dropping the product on the init
// cycle is a tempting shortcut and it silently loses one tap: the PE testbench
// caught exactly that, off by the first tap's product.)
//
// Because a 24-bit accumulator can never overflow the network (worst case
// 913k = 21 bit), integer addition is associative here, so the tap order is
// irrelevant -- any pipelining the engines do is numerically safe.
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module mac_lane #(
    parameter integer W_BITS     = `KWS_W_BITS,
    parameter integer ACT_BITS   = 8,
    parameter integer ACC_BITS   = 24,
    parameter integer SHIFT_BITS = 4
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // ---- job control -------------------------------------------------------
    input  wire                         en,        // this cycle's data belongs to the current job
    input  wire                         init,      // this tap is the job's first tap
    input  wire                         sel,       // accumulate bank: 0 = A, 1 = B

    // ---- data --------------------------------------------------------------
    input  wire signed [W_BITS-1:0]     w_i,       // INT8 weight code
    input  wire        [ACT_BITS-1:0]   a_i,       // UINT8 activation code

    // ---- per-lane constants (from the bias/shift ROM) ----------------------
    input  wire signed [ACC_BITS-1:0]   bias_i,    // folded bias, incl. 2**(S-1)
    input  wire        [SHIFT_BITS-1:0] shift_i,   // requantisation shift S

    // ---- read-out ----------------------------------------------------------
    input  wire                         rd_sel,    // which bank y_o shows
    output wire        [ACT_BITS-1:0]   y_o,       // clamp(acc >>> shift, 0, 255)
    // unclamped shifted value, low 16 bits -- the fully connected layer needs a
    // signed logit and must not be clamped to [0,255].  Safe to truncate: the
    // exporter checks that every logit fits in 16 bits (measured range is +-90).
    output wire signed [15:0]           ys_o,

    // ---- monitor (testbench / overflow check) ------------------------------
    output wire signed [ACC_BITS-1:0]   acc_a_o,
    output wire signed [ACC_BITS-1:0]   acc_b_o,
    output wire                         ovf_o      // 1 when |acc| >= 2**(ACC_BITS-1)
);

    localparam integer PROD_W = W_BITS + ACT_BITS;        // 16

    // -------------------------------------------------------------------------
    // Multiply.  `a_i` is unsigned, so it must be zero-extended into a *signed*
    // operand: a bare `w_i * a_i` would make the whole expression unsigned and
    // turn -1 into 255.  The 17-bit result is truncated to 16 because the true
    // product is bounded by 127*255 < 2^15.
    // -------------------------------------------------------------------------
    wire signed [PROD_W:0]        prod_wide = w_i * $signed({1'b0, a_i});
    wire signed [PROD_W-1:0]      prod      = prod_wide[PROD_W-1:0];
    wire signed [ACC_BITS-1:0]    prod_ext  = {{(ACC_BITS-PROD_W){prod[PROD_W-1]}}, prod};

    // -------------------------------------------------------------------------
    // Dual accumulator
    // -------------------------------------------------------------------------
    reg signed [ACC_BITS-1:0] acc_a, acc_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_a <= {ACC_BITS{1'b0}};
            acc_b <= {ACC_BITS{1'b0}};
        end else if (en) begin
            if (sel)
                acc_b <= (init ? bias_i : acc_b) + prod_ext;
            else
                acc_a <= (init ? bias_i : acc_a) + prod_ext;
        end
    end

    assign acc_a_o = acc_a;
    assign acc_b_o = acc_b;

    // -------------------------------------------------------------------------
    // Output stage: arithmetic right shift (== floor), ReLU, saturation.
    // ReLU needs no separate hardware: activations have zero point 0, so
    // clamping the negative side at 0 *is* the ReLU.
    // -------------------------------------------------------------------------
    wire signed [ACC_BITS-1:0] acc_rd  = rd_sel ? acc_b : acc_a;
    wire signed [ACC_BITS-1:0] shifted = acc_rd >>> shift_i;

    // Any set bit above bit 7 means "saturate"
    assign y_o = shifted[ACC_BITS-1]                  ? {ACT_BITS{1'b0}}
               : (|shifted[ACC_BITS-1:ACT_BITS])      ? {ACT_BITS{1'b1}}
               : shifted[ACT_BITS-1:0];

    assign ys_o = shifted[15:0];

    // -------------------------------------------------------------------------
    // Overflow monitor.  Cannot fire for the exported network (worst case is
    // 21 bit); it exists so a future model change is caught in simulation
    // instead of silently wrapping.
    // -------------------------------------------------------------------------
    assign ovf_o = (acc_a[ACC_BITS-1] ^ acc_a[ACC_BITS-2]) |
                   (acc_b[ACC_BITS-1] ^ acc_b[ACC_BITS-2]);

endmodule
