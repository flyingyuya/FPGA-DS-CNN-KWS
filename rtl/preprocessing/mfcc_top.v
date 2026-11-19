// =============================================================================
// mfcc_top.v -- the whole on-chip MFCC front end: 1 s of 16 kHz PCM in,
//               40 x 101 UINT8 codes out (FMEM write port).
//
//   sample buffer (16000 x 16)
//     -> per frame f (0..100): 512-point windowed frame in Q1.23
//          tap t < 480 : x = (samp[reflect(f*160 + t - 256)] * hann[t] + 64) >>> 7
//          tap t >= 480: x = 0
//     -> fft_core (N=512) -> 256 power bins (Q2.46)
//     -> mel_fb           -> 40 mel energies
//     -> log_q8           -> 40 logs (Q8.8)
//     -> dct40            -> 40 DCT coefficients (Q9.8)
//     -> code = clamp((A[c]*dct + B[c] + 2^19) >> 20, 0, 255) -> FMEM[c*101 + f]
//
// Reflect padding matches torch.stft(center=True, pad_mode="reflect"), so the map
// is what the model was trained on.
//
// IMPORTANT (this bit me three times in this project): every memory/ROM read that
// feeds a multiplier must be addressed COMBINATIONALLY from the loop counters, or
// by a register written at least two states earlier.  A register written in the
// same state that the memory samples hands the MAC the previous element -- which
// produced exactly ±1..2 LSB differences that look like rounding noise (see the
// mel, dct and sample-buffer fixes in docs/refactor-plan.md sec. 18-21).
//
// Cost: ~512 (feed, 2 cycles/tap) + 4608 (FFT) + 937 (mel) + 40 (log)
//       + 3202 (DCT) + 40 (writes) ~= 9,350 cycles per frame; x101 = 19.6 ms for
//       the whole map.  Sustained real-time only needs one frame per 10 ms hop.
// =============================================================================
`timescale 1ns / 1ps

module mfcc_top #(
    parameter integer NSAMP     = 16000,
    parameter integer NFRAME    = 101,
    parameter integer NFFT      = 512,
    parameter integer WIN       = 480,
    parameter integer HOP       = 160,
    parameter integer LOG2N     = 9,
    parameter integer NNORM     = 40,
    parameter integer NORM_FRAC = 20,
    parameter integer SKIP_LOAD = 0,
    parameter         MEM_DIR   = "mem"
) (
    input  wire                clk,
    input  wire                rst_n,

    // ---- sample stream (16-bit PCM, 16 kHz) --------------------------------
    input  wire                s_valid,
    input  wire signed [15:0]  s_data,
    output wire                s_ready,

    // ---- control -----------------------------------------------------------
    input  wire                start,
    output reg                 busy,
    output reg                 done,

    // ---- MFCC map out (FMEM write port) ------------------------------------
    output reg                 mfcc_we,
    output reg  [11:0]         mfcc_addr,
    output reg  [7:0]          mfcc_data,

    // ---- observability -----------------------------------------------------
    output reg  [31:0]         cycles_o,
    output reg  [7:0]          frame_o,

    // ---- debug: every stage's output for frame 0, one value per write ------
    // Added after several rounds of single-value probing proved useless for
    // locating a 26%-of-codes +-1..3 residue: the only reliable way is a whole
    // frame at every stage.  stage 0 = fft power (256), 1 = mel (40),
    // 2 = log (40), 3 = dct (40).
    output reg                 dbg_we,
    output reg  [1:0]          dbg_stage,
    output reg  [11:0]         dbg_addr,
    output reg  [47:0]         dbg_data,
    // stage snapshots for frame 0 (comparison against sim/reference_model's
    // intermediate values; a testbench cannot read the nested power buffer
    // reliably under iverilog)
    output reg                 dbg_taken, dbg_taken2,
    output reg  [47:0]         dbg_pw0,
    output reg  [53:0]         dbg_mel0,
    output reg  [15:0]         dbg_log0,
    output reg  [23:0]         dbg_dct0
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
    (* ram_style = "block" *) reg signed [15:0] samp [0:NSAMP-1];
    reg [13:0]        wr_ptr;
    // samp_re MUST be combinational: as a register it is high during S_WB, by
    // which time t_idx has already advanced, so samp_q ended up holding
    // samp[ridx(t+1)] -- a whole frame shifted by one sample.
    wire              samp_re = (state == S_WA);
    reg signed [15:0] samp_q;

    always @(posedge clk) begin
        if (s_valid && buf_ready) samp[wr_ptr] <= s_data;
        if (samp_re)              samp_q   <= samp[ridx];   // combinational address
    end
    assign s_ready = buf_ready;

    // =========================================================================
    // window ROM and the normalisation ROM (combinational reads)
    // =========================================================================
    wire [8:0]  win_addr = t_idx[8:0];   // combinational: the ROM is a comb read
                                        // and registered addresses have burned this
                                        // project six times
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
    always @(posedge clk) begin
        if (frame == 8'd0) begin
            if (mel_valid && mel_idx == 6'd0) dbg_mel0 <= mel_data;
            if (log_valid && n_idx == 6'd0)   dbg_log0 <= log_data;
            if (dct_valid && dct_idx == 6'd0) dbg_dct0 <= dct_data;
            // capture the FFT's own output for bins 0/1/2 (before the mel buffer),
            // and the mel buffer contents once everything is written
            if (fft_valid) begin
                if (bins == 9'd0 && !dbg_taken) begin dbg_pw0 <= fft_pow; dbg_taken <= 1'b1; end
                if (bins == 9'd1 && !dbg_taken2) begin dbg_mel0 <= {6'd0, fft_pow}; dbg_taken2 <= 1'b1; end
            end
            if (state == S_FFT && !fft_busy && bins >= NFFT/2)
                dbg_log0 <= u_mel.pw_mem[0][15:0];
        end
    end

    // ---------------------------------------------------------------------
    // DEBUG ONLY: shadow copy of the power bins + a post-hoc dump.
    //
    // Live probing of fft_pow / mel_data from a neighbouring always block kept
    // reading X (three separate rounds of it), so instead of sampling a stream
    // we keep a copy and walk it with a counter afterwards.  No valid/data
    // alignment, no "new value or old value" question.
    // Remove this block (and the shadow memory, ~1.5 BRAM) before tape-out.
    // ---------------------------------------------------------------------
    (* ram_style = "block" *) reg [47:0] pw_shadow [0:NFFT/2-1];
    always @(posedge clk) if (fft_valid) pw_shadow[fft_idx] <= fft_pow;

    reg        dmp_go;
    reg [8:0]  dmp_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_we <= 1'b0; dbg_stage <= 2'd0; dbg_addr <= 12'd0; dbg_data <= 48'd0;
            dmp_go <= 1'b0; dmp_cnt <= 9'd0;
        end else begin
            dbg_we <= 1'b0;
            if (frame == 8'd1 && mel_start) begin
                dmp_go  <= 1'b1;              // frame 0's mel has just been kicked off
                dmp_cnt <= 9'd0;
            end else if (dmp_go) begin
                dbg_we    <= 1'b1;
                dbg_stage <= 2'd0;
                dbg_addr  <= {4'd0, dmp_cnt[7:0]};
                dbg_data  <= pw_shadow[dmp_cnt[7:0]];
                dmp_cnt   <= dmp_cnt + 9'd1;
                if (dmp_cnt == NFFT/2 - 1) dmp_go <= 1'b0;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; buf_ready <= 1'b0;
            frame <= 8'd0; t_idx <= 10'd0; n_idx <= 6'd0; bins <= 9'd0;
            wr_ptr <= 14'd0; cyc <= 32'd0;
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

            case (state)
            S_IDLE: begin
                buf_ready <= 1'b0;
                if (start) begin
                    busy  <= 1'b1;
                    frame <= 8'd0;
                    cyc   <= 32'd0;
                    wr_ptr<= 14'd0;
                    state <= (SKIP_LOAD != 0) ? S_WA : S_LOAD;
                end
            end

            S_LOAD: begin
                buf_ready <= 1'b1;
                if (s_valid) begin
                    wr_ptr <= wr_ptr + 1'b1;
                    if (wr_ptr == NSAMP - 1) begin
                        buf_ready <= 1'b0;
                        state     <= S_WA;
                    end
                end
            end

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
