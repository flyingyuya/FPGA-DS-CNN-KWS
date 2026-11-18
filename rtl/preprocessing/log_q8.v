// =============================================================================
// log_q8.v -- natural log of a mel energy, Q8.8 output.
//
//   e       = highest set bit of (mel + eps)          0..53
//   m       = the 8 bits just below that bit          0..255
//   ln_q15  = e*LN2 + LUT[m] - 28*LN2                 (Q1.15)
//   log_q8  = (ln_q15 + 64) >>> 7                      (Q8.8)
//
// Why an 8-bit mantissa index is enough: the truncation error is at most
// 1/512 = 0.002 nats, which is half an LSB of the Q8.8 output, so a bigger table
// buys nothing (measured with the full front-end model: 92.77 % at 12 bits vs
// 92.87 % at 8 bits -- indistinguishable).
//
// The -28*LN2 constant folds two scaling factors into one adder:
//   * the mel energy is an integer in Q2.46, i.e. 2^46 times its float value;
//   * the FFT halves once per stage (/512), so the hardware power is 2^18 smaller
//     than the software one that the model was trained on.
// 46 - 18 = 28.  Getting this wrong shifts every log by a constant, which biases
// only c0 and is invisible in the other 39 coefficients -- the worst kind of bug.
//
// EPS = 1e-6 * 2^28 reproduces the software's log(mel + 1e-6).
// =============================================================================
`timescale 1ns / 1ps
`include "log_lut.v"

module log_q8 #(
    parameter integer MEL_BITS = 54,
    parameter integer LUT_BITS = 8,
    parameter integer EPS      = 268        // 1e-6 * 2^28
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          in_valid,
    input  wire [MEL_BITS-1:0]           mel_i,
    output reg                           out_valid,
    output reg  signed [15:0]            log_o       // Q8.8
);

    localparam [15:0] LN2_Q15 = 16'd22713;
    localparam integer       SHIFT_CONST = 28;
    localparam signed [24:0] CONST_Q15  = -(SHIFT_CONST * 22713);

    // ---- stage 1: add the epsilon, find the exponent and the mantissa ------
    wire [MEL_BITS:0] v = mel_i + EPS;                 // one extra bit for the add

    reg [5:0] e_c;                             // exponent, combinational
    reg [7:0] e_q, m_q;                        // stage-1 registers
    reg       v1;

    integer k;
    always @* begin
        e_c = 6'd0;
        for (k = 0; k <= MEL_BITS; k = k + 1)
            if (v[k]) e_c = k[5:0];                // last set bit wins = MSB
    end

    // bits [e-1 : e-8] of v, or a left shift when e < 8
    wire [MEL_BITS:0] shifted = (e_c >= LUT_BITS) ? (v >> (e_c - LUT_BITS))
                                                  : (v << (LUT_BITS - e_c));
    wire [7:0] mant = shifted[LUT_BITS-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1  <= 1'b0;
            e_q <= 8'd0;
            m_q <= 8'd0;
        end else begin
            v1  <= in_valid;
            e_q <= {2'b0, e_c};
            m_q <= mant;
        end
    end

    // ---- the mantissa table is combinational (256 x 16 LUTRAM) -------------
    wire [15:0] lut;
    log_mant_rom u_lut (.addr(m_q), .dout(lut));

    // ---- stage 2: combine and scale ---------------------------------------
    wire signed [24:0] ln_q15 = $signed({16'd0, e_q}) * $signed({9'd0, LN2_Q15})
                              + $signed({9'd0, lut}) + CONST_Q15;
    wire signed [24:0] rnd    = ln_q15 + 25'sd64;
    wire signed [24:0] q8     = rnd >>> 7;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            log_o     <= 16'sd0;
        end else begin
            out_valid <= v1;
            if (v1)
                log_o <= (q8 > 25'sd32767)  ? 16'sd32767 :
                         (q8 < -25'sd32768) ? -16'sd32768 :
                         q8[15:0];
        end
    end

endmodule
