// =============================================================================
// dct.v -- NxN DCT-II over the log-mel vector, one multiplier, time shared.
//
//     for c in 0..39:                      # output coefficient
//         acc = 0
//         for k in 0..39: acc += D[c][k] * log[k]
//         dct[c] = (acc + 2^14) >>> 15     # Q9.8
//
// 1600 MACs at two cycles each (address, then data) = ~3,200 cycles = 67 us per
// frame, i.e. 0.7 % of the 10 ms hop.  There is no reason to spend N multipliers
// on a NxN matrix that runs once per frame, and one multiplier cannot get the
// address/data alignment wrong.
//
// Formats
//   log[k] : 16 bit signed, Q8.8 (from log_q8)
//   D[c,k] : 16 bit signed, Q1.15 (mem/preprocessing/dct_coef.mem, row major)
//   acc    : 40 bit signed, Q9.23 -- the shift is done ONCE at the end, which is
//            different from the mel engine (it truncates per term).  The mel path
//            had precision headroom; this one feeds the classifier's
//            per-coefficient normalisation, where the high-order cepstral
//            coefficients have a small standard deviation and amplify errors.
//   dct[c] : 24 bit signed, Q9.8
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module dct #(
    parameter integer N        = `KWS_H_IN,   // N_MFCC == N_MELS
    parameter integer LOG_FRAC = 8,
    parameter integer DCT_BITS = 24,
    parameter integer ACC_BITS = 40,
    parameter         INIT_FILE = "mem/preprocessing/dct_coef.mem"
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- log-mel vector in (one value per cycle) ---------------------------
    input  wire                    log_we,
    input  wire [5:0]              log_addr,
    input  wire signed [15:0]      log_data,

    // ---- control -----------------------------------------------------------
    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    // ---- DCT coefficients out ---------------------------------------------
    output reg                     dct_valid,
    output reg  [5:0]              dct_idx,
    output reg  signed [DCT_BITS-1:0] dct_data
);

    localparam [1:0] S_IDLE = 2'd0, S_ADDR = 2'd1, S_MAC = 2'd2, S_DONE = 2'd3;

    // ---- log buffer: registers, so the read is combinational ---------------
    reg signed [15:0] logbuf [0:N-1];
    integer j;
    always @(posedge clk) begin
        if (log_we) logbuf[log_addr] <= log_data;
    end

    // ---- loop counters -----------------------------------------------------
    reg [5:0]  c_idx, k_idx;
    // COMBINATIONAL coefficient address: the BRAM latches it at the end of S_ADDR,
    // so a registered address would hand S_MAC the previous coefficient (the
    // failure is invisible on the c0 row, whose 40 coefficients are identical).
    wire [11:0] coef_addr = c_idx * N + k_idx;

    // ---- coefficient ROM: one BRAM, registered read -----------------------
    (* ram_style = "block" *) reg signed [15:0] coef_mem [0:N*N-1];
    reg signed [15:0] coef_q;
    always @(posedge clk) coef_q <= coef_mem[coef_addr];

    initial begin
        if (INIT_FILE != "") $readmemh(INIT_FILE, coef_mem);
    end

    reg [ACC_BITS-1:0] acc;
    wire signed [31:0] prod = coef_q * logbuf[k_idx];      // Q9.23
    wire [ACC_BITS-1:0] prod_ext = {{(ACC_BITS-32){prod[31]}}, prod};
    wire [ACC_BITS-1:0] acc_next = acc + prod_ext;

    wire signed [ACC_BITS-1:0] rounded = acc_next + {{(ACC_BITS-15){1'b0}}, 1'b1, 14'd0};
    wire signed [DCT_BITS-1:0] dct_out = rounded[15 +: DCT_BITS];

    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            c_idx     <= 6'd0;
            k_idx     <= 6'd0;
            acc       <= {ACC_BITS{1'b0}};
            dct_valid <= 1'b0;
            dct_idx   <= 6'd0;
            dct_data  <= {DCT_BITS{1'b0}};
        end else begin
            done      <= 1'b0;
            dct_valid <= 1'b0;

            case (state)
            S_IDLE: begin
                if (start) begin
                    busy      <= 1'b1;
                    state     <= S_ADDR;
                    c_idx     <= 6'd0;
                    k_idx     <= 6'd0;
                    acc       <= {ACC_BITS{1'b0}};
                        end
            end

            // coefficient address for (c,k); the BRAM output lands next cycle and
            // logbuf is combinational, so both inputs are aligned in S_MAC
            S_ADDR: begin
                state <= S_MAC;
            end

            S_MAC: begin
                if (k_idx == N - 1) begin
                    dct_valid <= 1'b1;
                    dct_idx   <= c_idx;
                    dct_data  <= dct_out;
                    acc       <= {ACC_BITS{1'b0}};
                    k_idx     <= 6'd0;
                    if (c_idx == N - 1) begin
                        c_idx <= 6'd0;
                        state <= S_DONE;
                    end else begin
                        c_idx <= c_idx + 6'd1;
                        state <= S_ADDR;
                    end
                end else begin
                    acc   <= acc_next;
                    k_idx <= k_idx + 6'd1;
                    state <= S_ADDR;
                end
            end

            S_DONE: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
