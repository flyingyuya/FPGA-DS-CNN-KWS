// =============================================================================
// mel_fb.v -- mel filterbank: 256 power bins -> 40 mel energies.
//
// The 40 triangular filters of this model are sparse and *contiguous*: filter f
// spans bins [lo_f, lo_f+len_f) with len 3..30, and every bin belongs to at most
// two filters.  So instead of tapping every bin on every cycle (which would need a
// 40-way accumulator with two accesses per cycle) the engine walks ONE filter at a
// time with a single multiplier:
//
//     for f in 0..39:
//         acc = 0
//         for i in 0..len_f-1:  acc += coef_f[i] * power[lo_f + i]
//         mel[f] = acc >> 15
//
// That is 468 MACs per frame; with the 2-cycle pipeline below it costs ~1000
// cycles against a 10 ms frame budget of 480,000, so nothing here must be fast.
//
// Formats
//   power[] : 48 bit unsigned Q2.46 (re*re + im*im of Q1.23 spectrum samples)
//   coef[]  : 16 bit UNSIGNED Q1.15, 1.0 = 32768.  Unsigned on purpose: every
//             triangular peak is exactly 1.0, and signed Q1.15 would clip it to
//             32767, biasing all 40 filters low.
//   mel[]   : MF_BITS unsigned on the same Q2.46 scale as the power bins (the >>15
//             cancels the Q1.15), i.e. the float model's mel energy times 2^46.
//
// Span ROM word: {base[9:0], len[6:0], lo[7:0]}  (25 bit)
//
// Pipeline (one MAC every cycle once running)
//   cycle t   : issue pw/coef addresses for coefficient i
//   cycle t+1 : both memories' registered outputs are valid -> multiply, accumulate
// =============================================================================
`timescale 1ns / 1ps
`include "mel_rom.v"

module mel_fb #(
    parameter integer NBIN    = 256,
    parameter integer NFILT   = 40,
    parameter integer MF_BITS = 54          // mel accumulator width
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- power spectrum in (from fft_pow) ---------------------------------
    input  wire                 pw_we,
    input  wire [7:0]           pw_addr,
    input  wire [47:0]          pw_data,

    // ---- control -----------------------------------------------------------
    input  wire                 start,       // pulse once the 256 bins arrived
    output reg                  busy,
    output reg                  done,

    // ---- mel energies out --------------------------------------------------
    output reg                  mel_valid,
    output reg  [5:0]           mel_idx,
    output reg  [MF_BITS-1:0]   mel_data
);

    // Two cycles per MAC.  The addresses are COMBINATIONAL from the registered
    // (f_idx, i_idx) counters, so the memories latch coefficient i at the end of
    // S_ADDR and S_MAC consumes it -- no address register, hence no off-by-one
    // between the coefficient and the power bin.  (With registered addresses you
    // need three states per MAC for the same guarantee.)
    localparam [1:0] S_IDLE = 2'd0, S_ADDR = 2'd1, S_MAC = 2'd2, S_DONE = 2'd3;

    // ---- counters ----------------------------------------------------------
    reg  [5:0]  f_idx;                         // filter being accumulated
    reg  [8:0]  i_idx;                         // coefficient index inside it

    // ---- power buffer: 1 write port + 1 registered read port ---------------
    (* ram_style = "block" *) reg [47:0] pw_mem [0:NBIN-1];
    reg [47:0] pw_q;

    // ---- filter span / coefficient ROMs -----------------------------------
    wire [24:0] span;                          // {base[9:0], len[6:0], lo[7:0]}
    wire [15:0] coef;
    reg  [15:0] coef_q;

    wire [9:0] base_f = span[24:15];
    wire [6:0] len_f  = span[14:8];
    wire [7:0] lo_f   = span[7:0];

    wire [8:0] raddr_coef = base_f[8:0] + i_idx;
    wire [7:0] raddr_bin  = lo_f + i_idx[7:0];

    mel_span_rom u_span (.addr(f_idx), .dout(span));
    mel_coef_rom u_coef (.addr(raddr_coef), .dout(coef));
    always @(posedge clk) coef_q <= coef;      // registered ROM read

    always @(posedge clk) begin
        if (pw_we) pw_mem[pw_addr] <= pw_data;
        pw_q <= pw_mem[raddr_bin];
    end

    wire [63:0] prod = coef_q * pw_q;          // 16 x 48 unsigned
    wire [48:0] term = prod >> 15;             // back to the Q2.46 power scale
    wire [MF_BITS-1:0] term_ext = {{(MF_BITS-49){1'b0}}, term};

    reg [1:0]  state;
    reg [MF_BITS-1:0] acc;
    wire [MF_BITS-1:0] acc_next = acc + term_ext;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            f_idx     <= 6'd0;
            i_idx     <= 9'd0;
            acc       <= {MF_BITS{1'b0}};
            mel_valid <= 1'b0;
            mel_idx   <= 6'd0;
            mel_data  <= {MF_BITS{1'b0}};
        end else begin
            done      <= 1'b0;
            mel_valid <= 1'b0;

            case (state)
            S_IDLE: begin
                if (start) begin
                    busy  <= 1'b1;
                    state <= S_ADDR;
                    f_idx <= 6'd0;
                    i_idx <= 9'd0;
                    acc   <= {MF_BITS{1'b0}};
                end
            end

            // the memories latch the combinational address set by (f_idx, i_idx)
            S_ADDR: begin
                state <= S_MAC;
            end

            S_MAC: begin
                if ({2'b0, i_idx} + 10'd1 >= {3'b0, len_f}) begin
                    mel_valid <= 1'b1;
                    mel_idx   <= f_idx;
                    mel_data  <= acc_next;
                    acc       <= {MF_BITS{1'b0}};
                    i_idx     <= 9'd0;
                    if (f_idx == NFILT - 1) begin
                        f_idx <= 6'd0;
                        state <= S_DONE;
                    end else begin
                        f_idx <= f_idx + 6'd1;
                        state <= S_ADDR;
                    end
                end else begin
                    acc   <= acc_next;
                    i_idx <= i_idx + 9'd1;
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
