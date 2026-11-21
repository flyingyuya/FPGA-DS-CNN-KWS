// =============================================================================
// mfcc_top.v -- the whole on-chip MFCC front end: 1 s of 16 kHz PCM in,
//               KWS_H_IN x KWS_W_IN UINT8 codes out (FMEM write port).
//
//   sample buffer (16000 x 16)
//     -> per frame f (0..100): 512-point windowed frame in Q1.23
//          tap t < 480 : x = (samp[reflect(f*160 + t - 256)] * hann[t] + 64) >>> 7
//          tap t >= 480: x = 0
//     -> fft_core (N=512) -> 256 power bins (Q2.46)
//     -> mel_fb           -> NNORM mel energies
//     -> log_q8           -> NNORM logs (Q8.8)
//     -> dct40            -> NNORM DCT coefficients (Q9.8)
//     -> code = clamp((A[c]*dct + B[c] + 2^19) >> 20, 0, 255) -> FMEM[c*101 + f]
//
// Reflect padding matches torch.stft(center=True, pad_mode="reflect"), so the map
// is what the model was trained on.
//
// Cost scales with NNORM: at NNORM=20 a frame is ~7.6k cycles, so the whole
// map (NFRAME=101) takes 768,913 cycles = 16.0 ms at 48 MHz.  Sustained real-time only needs one frame per 10 ms hop.
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module mfcc_top #(
    parameter integer NSAMP     = 16000,          // 1 s at 16 kHz (model input spec)
    parameter integer NFRAME    = `KWS_W_IN,
    parameter integer NFFT      = 512,
    parameter integer WIN       = 480,
    parameter integer HOP       = 160,
    parameter integer LOG2N     = 9,
    parameter integer NNORM     = `KWS_H_IN,
    parameter integer NORM_FRAC = 20,
    parameter integer SKIP_LOAD = 0,
    parameter         MEM_DIR   = "mem"
) (
    input  wire                clk,
    input  wire                rst_n,

    // ---- sample stream (16-bit PCM, 16 kHz) --------------------------------
    input  wire                s_valid,
    input  wire signed [15:0]  s_data,
    input  wire                freeze,  // 1 = stop capturing (a run is in progress)
    output wire                s_ready,

    // ---- control -----------------------------------------------------------
    input  wire                start,
    output reg                 busy,
    output reg                 done,

    // ---- MFCC map out (FMEM write port) ------------------------------------
    output reg                 mfcc_we,
    output reg  [`KWS_IN_ADDR_W-1:0] mfcc_addr,
    output reg  [7:0]          mfcc_data,

    // ---- observability -----------------------------------------------------
    output reg  [31:0]         cycles_o,
    output reg  [7:0]          frame_o
);

    localparam [3:0] S_IDLE = 4'd0, S_LOAD = 4'd1, S_WA = 4'd2,  S_WB   = 4'd3,
                     S_FFT  = 4'd4, S_MEL  = 4'd5, S_LOG  = 4'd6,  S_DCT  = 4'd7,
                     S_NORM = 4'd8, S_NDONE= 4'd9, S_DONE = 4'd10,
                     S_MEL2 = 4'd11;

    // =========================================================================
    // counters and the combinational sample index
    // =========================================================================
    reg [3:0]  state;
    reg        buf_ready;
    reg [7:0]  frame;
    reg [9:0]  t_idx;                        // tap 0..511
    reg [5:0]  n_idx;                        // mel / log / dct element
    reg [8:0]  bins;
    reg [31:0] cyc;

    wire signed [31:0] fbase = $signed({24'd0, frame}) * HOP;
    wire signed [31:0] widx  = fbase + $signed({22'd0, t_idx}) - (NFFT/2);
    wire [31:0]        widx_abs = widx[31] ? (32'd0 - widx) : widx;
    wire [31:0]        widx_ref = (widx >= NSAMP) ? (2*NSAMP - 2 - widx) : widx_abs;
    wire [13:0]        ridx = widx_ref[13:0];

    // =========================================================================
    // sample buffer
    // =========================================================================
    // The sample buffer is a 16000-entry RING, written continuously and frozen
    // while a run is in progress.  Freezing costs the ~21 ms of audio thatarrives 
    // during the MFCC run, shrinks as the MFCC latency is reduced.
    //
    // `base` is the write pointer latched at `start`, i.e. the oldest sample;
    // reads are offset by it and wrap modulo NSAMP.
    (* ram_style = "block" *) reg signed [15:0] samp [0:NSAMP-1];
    reg [13:0]        wr_ptr;
    reg [13:0]        base;
    // samp_re MUST be combinational: as a register it is high during S_WB, by
    // which time t_idx has already advanced, so samp_q ended up holding samp[ridx(t+1)].
    wire              samp_re = (state == S_WA);
    reg signed [15:0] samp_q;

    // ring offset: (base + reflected index) modulo NSAMP
    wire [14:0] ridx_sum  = {1'b0, base} + {1'b0, ridx};
    wire [13:0] ridx_ring = (ridx_sum >= NSAMP) ? (ridx_sum - NSAMP) : ridx_sum[13:0];

    // memory write stays out of the reset branch so Vivado can infer BRAM
    always @(posedge clk) begin
        if (s_valid && !freeze) samp[wr_ptr] <= s_data;
        if (samp_re)            samp_q <= samp[ridx_ring];
    end
    assign s_ready = ~freeze;

    // =========================================================================
    // window ROM and the normalisation ROM (combinational reads)
    // =========================================================================
    wire [8:0]  win_addr = t_idx[8:0];   // the ROM MUST be a combinational read
    wire [15:0] win_q;
    window_rom u_win (.addr(win_addr), .dout(win_q));

    reg  [5:0]  norm_addr;
    wire [55:0] norm_q;
    mfcc_norm_rom u_norm (.addr(norm_addr), .dout(norm_q));
    wire signed [23:0] norm_a = norm_q[23:0];
    wire signed [31:0] norm_b = norm_q[55:24];

    // =========================================================================
    // FFT
    // =========================================================================
    reg  signed [23:0] fft_in;
    reg                fft_we;
    wire [LOG2N-2:0]   fft_idx;
    wire [48:0]        fft_pow_full;
    wire               fft_valid, fft_busy;
    wire [7:0]         fft_drop;
    wire [47:0]        fft_pow = fft_pow_full[47:0];

    wire signed [23:0] wprod = (samp_q * $signed({1'b0, win_q}) + 64) >>> 7;

    fft_core #(.N(NFFT), .LOG2N(LOG2N), .DW(24), .TW(16)) u_fft (
        .clk(clk), .rst_n(rst_n), .x_in(fft_in), .x_valid(fft_we),
        .y_index(fft_idx), .y_pow(fft_pow_full), .y_valid(fft_valid),
        .busy(fft_busy), .drop_cnt(fft_drop));

    // =========================================================================
    // mel -> log -> dct
    // =========================================================================
    reg                mel_start;
    wire               mel_busy, mel_done, mel_valid;
    wire [5:0]         mel_idx;
    wire [53:0]        mel_data;

    mel_fb u_mel (
        .clk(clk), .rst_n(rst_n),
        // ungated: the mel engine only reads this buffer in S_MEL/S_LOG, so
        // writing it whenever the FFT produces a bin is safe -- and gating on
        // state == S_FFT silently drops any bin that arrives on the transition
        .pw_we(fft_valid), .pw_addr(fft_idx), .pw_data(fft_pow),
        .start(mel_start), .busy(mel_busy), .done(mel_done),
        .mel_valid(mel_valid), .mel_idx(mel_idx), .mel_data(mel_data));

    reg                log_we;
    reg  [53:0]        log_mel;
    wire               log_valid;
    wire signed [15:0] log_data;

    log_q8 u_log (
        .clk(clk), .rst_n(rst_n), .in_valid(log_we), .mel_i(log_mel),
        .out_valid(log_valid), .log_o(log_data));

    reg                dct_log_we;
    reg  [5:0]         dct_log_addr;
    reg                dct_start;
    wire               dct_busy, dct_done, dct_valid;
    wire [5:0]         dct_idx;
    wire signed [23:0] dct_data;

    dct40 u_dct (
        .clk(clk), .rst_n(rst_n),
        .log_we(dct_log_we), .log_addr(dct_log_addr), .log_data(log_data),
        .start(dct_start), .busy(dct_busy), .done(dct_done),
        .dct_valid(dct_valid), .dct_idx(dct_idx), .dct_data(dct_data));

    // =========================================================================
    // normalise: one multiply-add folds the per-coefficient scale, the zero
    // offset and the UINT8 quantisation
    // =========================================================================
    reg signed [23:0] p_data;
    reg  [5:0]        p_idx;
    reg               p_valid;

    wire signed [47:0] norm_prod = norm_a * p_data;
    wire signed [56:0] norm_sum  = {{9{norm_prod[47]}}, norm_prod}
                                 + {{25{norm_b[31]}}, norm_b};
    wire signed [56:0] norm_sh   = (norm_sum + (1 << (NORM_FRAC-1))) >>> NORM_FRAC;

    // =========================================================================
    // FSM
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; buf_ready <= 1'b0;
            frame <= 8'd0; t_idx <= 10'd0; n_idx <= 6'd0; bins <= 9'd0;
            wr_ptr <= 14'd0; base <= 14'd0; cyc <= 32'd0;
            fft_we <= 1'b0; fft_in <= 24'sd0;
            mel_start <= 1'b0;
            log_we <= 1'b0; log_mel <= 54'd0;
            dct_log_we <= 1'b0; dct_log_addr <= 6'd0; dct_start <= 1'b0;
            norm_addr <= 6'd0; p_data <= 24'sd0; p_idx <= 6'd0; p_valid <= 1'b0;
            mfcc_we <= 1'b0; mfcc_addr <= 12'd0; mfcc_data <= 8'd0;
            cycles_o <= 32'd0; frame_o <= 8'd0;
        end else begin
            done      <= 1'b0;
            fft_we    <= 1'b0;
            mel_start <= 1'b0;
            log_we    <= 1'b0;
            dct_log_we<= 1'b0;
            dct_start <= 1'b0;
            mfcc_we   <= 1'b0;
            if (busy) cyc <= cyc + 1'b1;
            if (s_valid && !freeze)
                wr_ptr <= (wr_ptr == NSAMP - 1) ? 14'd0 : wr_ptr + 14'd1;

            case (state)
            S_IDLE: begin
                buf_ready <= 1'b0;
                if (start) begin
                    busy  <= 1'b1;
                    frame <= 8'd0;
                    cyc   <= 32'd0;
                    base  <= wr_ptr;          // oldest sample = where the next
                                              // write would have gone
                    state <= (SKIP_LOAD != 0) ? S_WA : S_LOAD;
                end
            end

            // with a ring there is nothing to load: the buffer is already full
            S_LOAD: state <= S_WA;

            // issue the sample read and the window address for tap t_idx
            S_WA: begin
                frame_o  <= frame;
                state    <= S_WB;
            end

            // multiply, push to the FFT
            S_WB: begin
                fft_we <= 1'b1;
                fft_in <= (t_idx < WIN) ? wprod : 24'sd0;
                t_idx  <= t_idx + 1'b1;
                if (t_idx == NFFT - 1) begin
                    t_idx <= 10'd0;
                    bins  <= 9'd0;
                    state <= S_FFT;
                end else begin
                    state <= S_WA;
                end
            end

            // collect the 256 power bins into the mel engine's buffer
            S_FFT: begin
                if (fft_valid) bins <= bins + 1'b1;
                if (!fft_busy && bins >= NFFT/2) state <= S_MEL;
            end

            // two cycles: the last power-bin write in S_FFT lands on the edge that
            // enters S_MEL, so give the buffer one full cycle to settle before the
            // mel engine reads it (otherwise every filter sees the same stale data)
            S_MEL: begin
                mel_start <= 1'b1;
                n_idx     <= 6'd0;
                state     <= S_MEL2;
            end

            S_MEL2: begin
                state <= S_LOG;
            end

            // 40 mel energies in, 40 logs out, straight into the DCT's buffer
            S_LOG: begin
                if (mel_valid) begin
                    log_we  <= 1'b1;
                    log_mel <= mel_data;
                end
                if (log_valid) begin
                    dct_log_we   <= 1'b1;
                    dct_log_addr <= n_idx;
                    n_idx        <= n_idx + 1'b1;
                    if (n_idx == NNORM - 1) state <= S_DCT;
                end
            end

            S_DCT: begin
                dct_start <= 1'b1;
                state     <= S_NORM;
            end

            S_NORM: begin
                p_valid <= dct_valid;
                if (dct_valid) begin
                    norm_addr <= dct_idx;
                    p_data    <= dct_data;
                    p_idx     <= dct_idx;
                end
                if (p_valid) begin
                    mfcc_we   <= 1'b1;
                    mfcc_addr <= p_idx * NFRAME + frame;
                    mfcc_data <= (norm_sh > 57'sd255) ? 8'd255 :
                                 (norm_sh < 57'sd0)   ? 8'd0   : norm_sh[7:0];
                    if (p_idx == NNORM - 1) state <= S_NDONE;
                end
            end

            S_NDONE: begin
                if (!dct_busy) begin
                    if (frame == NFRAME - 1) begin
                        state <= S_DONE;
                    end else begin
                        frame <= frame + 8'd1;
                        state <= S_WA;
                    end
                end
            end

            S_DONE: begin
                busy     <= 1'b0;
                done     <= 1'b1;
                cycles_o <= cyc;
                state    <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
