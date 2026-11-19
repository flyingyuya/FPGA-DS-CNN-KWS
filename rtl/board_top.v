// =============================================================================
// board_top.v -- PA-StarLite board wrapper (xc7a100t-2fgg484).
//
// Scope: make the accelerator placeable and observable on the real board, using
// the pin set published in the board's own example projects.  The MFCC frame is
// baked into the FMEM block RAM at configuration time, so this build classifies
// one fixed test vector per key press:
//
//     key (W21)  -> start          one inference, 55,522 cycles (1.16 ms)
//     led[0]     -> busy           on while the accelerator runs
//     led[1]     -> result         blinks when a new class has been latched
//     uart_tx    -> 115200 8N1     sends 0xA5, class, margin in one burst
//
// M7 wired the real audio path:
//
//   WM8960 (I2S master, 16 kHz) --bclk/lrclk/adcdat--> i2s_rx
//        -> mfcc_top (16000 samples in, 40x101 INT8 map out, ~21 ms)
//        -> 4 KiB staging buffer (an FMEM-shaped map, c*101+f order)
//        -> accelerator_top's stream loader (skip_load=0)
//        -> UART (0xA5, class, margin)
//
// The staging buffer exists because mfcc_top emits values in frame-major order
// (40 coefficients per frame) while the loader wants address order (all 101
// frames of coefficient 0, then coefficient 1, ...).  Two clocks of BRAM is a
// lot cheaper than making the accelerator's address generator wrap modulo 101.
//
// Flow: power-up runs the I2C init; then each key press records one second of
// audio and classifies it.
//
// The key is synchronised and edge detected; a mechanical press is several
// milliseconds, longer than one inference, so the edge detector alone is enough
// and no debounce counter is needed.
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module board_top #(
    parameter integer NUM_CLASSES = `KWS_NUM_CLASSES,
    parameter         MEM_DIR     = "mem",
    parameter integer WM_DLY_MS   = 1        // gap between I2C register writes
) (
    input  wire sys_clk_p,
    input  wire sys_clk_n,
    input  wire sys_rstn,
    input  wire key,             // W21
    output wire uart_tx,         // P15
    output wire [1:0] led,       // W22 / Y22

    // ---- WM8960 (audio codec, I2S master) ---------------------------------
    input  wire aud_bclk,
    input  wire aud_lrclk,
    input  wire aud_adcdat,
    output wire aud_scl,
    inout  wire aud_sda
);


    // ---- clocking ----------------------------------------------------------
    wire clk, rst_n, locked;
    clk_gen u_clk (
        .clk_p(sys_clk_p), .clk_n(sys_clk_n),
        .rst_async_n(sys_rstn),
        .clk(clk), .rst_n(rst_n), .locked(locked)
    );

    // ---- key: synchronise, DEBOUNCE, then edge detect ----------------------
    // The original version had no debounce, on the theory that a mechanical press
    // lasts many milliseconds.  On the real board the key pin floats when it is
    // not driven and picks up noise, which produced a stream of phantom presses
    // and a continuous stream of UART reports.
    reg [2:0]  ks;
    reg        key_db;
    reg [13:0] dbc;                            // 2^14 / 48 MHz = 341 us
    reg        key_db_q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ks <= 3'b000; key_db <= 1'b0; dbc <= 14'd0; key_db_q <= 1'b0;
        end else begin
            ks <= {ks[1:0], key};
            if (ks[2] != key_db) begin
                if (dbc == 14'h3FFF) begin key_db <= ks[2]; dbc <= 14'd0; end
                else                        dbc    <= dbc + 14'd1;
            end else begin
                dbc <= 14'd0;
            end
            key_db_q <= key_db;
        end
    end

    wire key_edge = key_db & ~key_db_q;       // rising edge of the debounced key

    // Cool-down: one run takes ~1.02 s (1 s of audio + 22 ms of compute), so a
    // press that arrives sooner can only be noise.  2^26 / 48 MHz = 1.4 s.
    reg [25:0] lock;
    wire key_press = key_edge & (lock == 26'd0);


    // ---- WM8960 register init (I2C) + I2S receive --------------------------
    reg        wm_go;
    wire       wm_done;
    wire        aud_valid;
    wire signed [15:0] aud_data;

    i2s_rx u_i2s (
        .clk(clk), .rst_n(rst_n),
        .bclk(aud_bclk), .lrclk(aud_lrclk), .adc_dat(aud_adcdat),
        .s_valid(aud_valid), .s_data(aud_data));

    WM8960_init #(.CLK_FREQ_HZ(48_000_000), .DLY_MS(WM_DLY_MS)) u_wm (
        .Clk(clk), .Rst_n(rst_n), .Go(wm_go), .device_id(8'h34), .src_sel(1'b0),
        .Init_Done(wm_done), .i2c_sclk(aud_scl), .i2c_sdat(aud_sda));

    // ---- MFCC front end ----------------------------------------------------
    wire        mfcc_busy, mfcc_done;
    wire        mfcc_start;
    wire        mfcc_we;
    wire [11:0] mfcc_addr;
    wire [7:0]  mfcc_data;

    mfcc_top #(.SKIP_LOAD(0), .MEM_DIR(MEM_DIR)) u_mfcc (
        .clk(clk), .rst_n(rst_n),
        .s_valid(aud_valid), .s_data(aud_data), .s_ready(),
        .start(mfcc_start), .busy(mfcc_busy), .done(mfcc_done),
        .mfcc_we(mfcc_we), .mfcc_addr(mfcc_addr), .mfcc_data(mfcc_data),
        .cycles_o(), .frame_o());

    // ---- staging buffer for the accelerator's stream loader ----------------
    localparam integer NBYTES = 40 * 101;

    (* ram_style = "block" *) reg [7:0] map_buf [0:4095];
    always @(posedge clk) if (mfcc_we) map_buf[mfcc_addr] <= mfcc_data;

    // ---- sequencer state (declared here because the accelerator instance
    //      below consumes these signals) -------------------------------------
    localparam [2:0] S_INIT = 3'd0, S_KEY = 3'd1, S_MFCC = 3'd2,
                     S_STREAM = 3'd3, S_RUN = 3'd4;

    reg [2:0]  seq;
    reg        wm_started;
    reg        wm_ok;                 // codec acknowledged its I2C init
    reg [23:0] wm_to;                 // init timeout (~350 ms), see S_INIT
    reg        acc_start;
    reg [11:0] sptr;

    wire stream_valid = (seq == S_STREAM) && (sptr < NBYTES);
    wire stream_last  = (sptr == NBYTES - 1);
    wire stream_ready;

    // ---- accelerator -------------------------------------------------------
    wire        busy, done, result_valid;
    wire [NUM_CLASSES*16-1:0] logits;
    wire [3:0]  class_o;
    wire signed [15:0] max_o, margin_o;
    wire [31:0] cycles_o;
    wire [3:0]  layer_o;

    accelerator_top #(.NUM_CLASSES(NUM_CLASSES), .MEM_DIR(MEM_DIR)) u_accel (
        .clk(clk), .rst_n(rst_n),
        .start(acc_start),
        .skip_load(1'b0),                     // stream the MFCC map in
        .busy(busy), .done(done),
        .in_valid(stream_valid), .in_data(map_buf[sptr[11:0]]),
        .in_last(stream_last), .in_ready(stream_ready),
        .logits_o(logits), .class_o(class_o), .max_o(max_o), .margin_o(margin_o),
        .result_valid(result_valid), .cycles_o(cycles_o), .layer_o(layer_o));

    // ---- key cool-down (needs busy from the accelerator instance above) ------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) lock <= 26'd0;
        else if (key_press && seq == S_KEY && !mfcc_busy && !busy)
            lock <= 26'h3FF_FFFF;
        else if (lock != 26'd0)
            lock <= lock - 26'd1;
    end
    // =========================================================================
    // sequencer
    //   S_INIT   wait for the codec's I2C init
    //   S_KEY    idle until a key press
    //   S_MFCC   mfcc_top loads 16000 samples from i2s_rx, then runs (~21 ms)
    //   S_STREAM push the 4040-byte map into the accelerator's stream loader
    //   S_RUN    wait for the accelerator, then report
    // =========================================================================
    assign mfcc_start = (seq == S_KEY) && key_press;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seq        <= S_INIT;
            wm_go      <= 1'b0;
            wm_started <= 1'b0;
            wm_ok      <= 1'b0;
            wm_to      <= 24'hFF_FFFF;
            acc_start <= 1'b0;
            sptr      <= 12'd0;
        end else begin
            acc_start <= 1'b0;
            case (seq)
            // WM8960_init treats Go as a LEVEL with priority over the table
            // counter (else if (Go) cnt <= 0), so holding it high freezes the
            // sequence at zero writes.  It has to be a one-cycle pulse.
            S_INIT: begin
                if (wm_to != 24'd0) wm_to <= wm_to - 24'd1;
                if (wm_done) begin
                    seq   <= S_KEY;
                    wm_go <= 1'b0;
                    wm_ok <= 1'b1;
                end else if (!wm_started) begin
                    wm_go      <= 1'b1;
                    wm_started <= 1'b1;
                end else begin
                    wm_go <= 1'b0;
                    // Do not let a silent codec brick the demo: after ~350 ms
                    // (the real init takes ~70 ms at 400 kHz SCL) keep going and
                    // show the failure on led[1] instead of hanging forever.
                    if (wm_to == 24'd0) seq <= S_KEY;
                end
            end
            S_KEY: if (key_press && !mfcc_busy && !busy) seq <= S_MFCC;
            S_MFCC: if (mfcc_done) begin
                acc_start <= 1'b1;
                sptr      <= 12'd0;
                seq       <= S_STREAM;
            end
            S_STREAM: if (stream_ready && stream_valid) begin
                sptr <= sptr + 12'd1;
                if (sptr == NBYTES - 1) seq <= S_RUN;
            end
            S_RUN: if (result_valid) seq <= S_KEY;
            default: seq <= S_INIT;
            endcase
        end
    end

    // ---- result reporting --------------------------------------------------
    reg  [3:0]  class_q;
    reg         new_result;
    reg  [1:0]  tx_state;
    reg  [7:0]  tx_byte;
    reg         tx_send;
    wire        tx_busy;

    localparam [1:0] T_IDLE = 2'd0, T_CLASS = 2'd1, T_MARGIN = 2'd2, T_WAIT = 2'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_q    <= 4'd0;
            new_result <= 1'b0;
            tx_state   <= T_IDLE;
            tx_byte    <= 8'd0;
            tx_send    <= 1'b0;
        end else begin
            tx_send <= 1'b0;
            if (result_valid) begin
                class_q    <= class_o;
                new_result <= ~new_result;    // toggles on every new classification
            end
            case (tx_state)
            T_IDLE: if (result_valid) begin
                tx_byte  <= 8'hA5;
                tx_send  <= 1'b1;
                tx_state <= T_CLASS;
            end
            T_CLASS: if (!tx_busy && !tx_send) begin
                tx_byte  <= {4'h0, class_q};
                tx_send  <= 1'b1;
                tx_state <= T_MARGIN;
            end
            T_MARGIN: if (!tx_busy && !tx_send) begin
                tx_byte  <= margin_o[7:0];
                tx_send  <= 1'b1;
                tx_state <= T_WAIT;
            end
            T_WAIT: if (!tx_busy && !tx_send) tx_state <= T_IDLE;
            endcase
        end
    end

    uart_tx u_uart (
        .clk(clk), .rst_n(rst_n), .send(tx_send), .tx_data(tx_byte),
        .tx(uart_tx), .busy(tx_busy));

    // ---- bring-up indicators -----------------------------------------------
    // led[0] is a free-running heartbeat: if it is not blinking, the clock or
    // the reset never came up (bitstream not loaded, MMCM unlocked, wrong pins).
    // led[1] tells you where the sequencer is:
    //     fast blink : waiting for the codec's I2C init (or it timed out)
    //     solid      : computing (MFCC or classification)
    //     short flash: a new classification arrived
    //     off        : ready, press the key
    reg [24:0] hb;
    reg [22:0] hold;                  // ~175 ms result flash at 48 MHz

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hb   <= 25'd0;
            hold <= 23'd0;
        end else begin
            hb <= hb + 25'd1;
            if (result_valid)      hold <= 23'h7F_FFFF;
            else if (hold != 23'd0) hold <= hold - 23'd1;
        end
    end

    assign led[0] = hb[24];                                   // ~1.4 Hz
    assign led[1] = !wm_ok            ? hb[21] :              // ~5.7 Hz: codec not configured
                    (mfcc_busy | busy) ? 1'b1  :              // computing
                    (hold != 23'd0)    ? 1'b1  :              // new result
                                         1'b0;

endmodule
