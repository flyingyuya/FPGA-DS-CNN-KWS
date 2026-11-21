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
// wired the real audio path:
//
//   WM8960 (I2S master, 16 kHz) --bclk/lrclk/adcdat--> i2s_rx
//        -> mfcc_top (16000 samples in, 40x101 INT8 map out, ~21 ms)
//        -> FMEM, written directly through accelerator_top's map_we port
//        -> accelerator_top (skip_load=1)
//        -> UART (0xA5, class, margin)
//
// mfcc_top writes in frame-major order and the stream loader wants address
// order, but the two writers never overlap, so they share FMEM's write port
// behind a mux instead of going through a 4 KiB staging buffer.
//
// Flow: power-up runs the I2C init; then each key press records one second of
// audio and classifies it.
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

    // ---- key: debounce (20 ms) + single-shot pulse -------------------------
    wire key_press;

    key_debounce #(.CLK_HZ(48_000_000), .MS(20)) u_key (
        .clk(clk), .rst_n(rst_n),
        .key_n(key), .key_pressed(), .press(key_press));

    // ---- WM8960 register init (I2C) + I2S receive --------------------------
    reg wm_go;
    wire wm_done;
    wire aud_valid;
    wire signed [15:0] aud_data;

    i2s_rx u_i2s (
        .clk(clk), .rst_n(rst_n),
        .bclk(aud_bclk), .lrclk(aud_lrclk), .adc_dat(aud_adcdat),
        .s_valid(aud_valid), .s_data(aud_data));

    WM8960_init #(.CLK_FREQ_HZ(48_000_000), .DLY_MS(WM_DLY_MS)) u_wm (
        .Clk(clk), .Rst_n(rst_n), .Go(wm_go), .device_id(8'h34), .src_sel(1'b0),
        .Init_Done(wm_done), .i2c_sclk(aud_scl), .i2c_sdat(aud_sda));

    // ---- MFCC front end ----------------------------------------------------
    wire                        mfcc_busy, mfcc_done;
    reg                         mfcc_start;
    wire                        mfcc_we;
    wire [`KWS_IN_ADDR_W-1:0]   mfcc_addr;
    wire [7:0]                  mfcc_data;

    mfcc_top #(.SKIP_LOAD(0), .MEM_DIR(MEM_DIR),
               .NFRAME(`KWS_W_IN), .NNORM(`KWS_H_IN)) u_mfcc (
        .clk(clk), .rst_n(rst_n),
        .s_valid(aud_valid), .s_data(aud_data),
        .freeze(mfcc_busy),             // freeze the ring only while the MFCC runs
        .s_ready(),
        .start(mfcc_start), .busy(mfcc_busy), .done(mfcc_done),
        .mfcc_we(mfcc_we), .mfcc_addr(mfcc_addr), .mfcc_data(mfcc_data),
        .cycles_o(), .frame_o());

    // ---- sequencer state ---------------------------------------------------
    localparam [2:0] S_INIT = 3'd0, S_KEY = 3'd1, S_MFCC = 3'd2, S_RUN = 3'd3;

    reg [2:0]  seq;
    reg        wm_started;
    reg        wm_ok;                 // codec acknowledged its I2C init
    reg [23:0] wm_to;                 // init timeout (~350 ms), see S_INIT
    reg        acc_start;

    // ---- accelerator -------------------------------------------------------
    wire        act_busy, act_done;
    wire [NUM_CLASSES*16-1:0] logits;
    wire [3:0]  class_o;
    wire signed [15:0] max_o, margin_o;
    wire [31:0] cycles_o;
    wire [3:0]  layer_o;

    accelerator_top #(.NUM_CLASSES(NUM_CLASSES), .MEM_DIR(MEM_DIR)) u_accel (
        .clk(clk), .rst_n(rst_n),
        .start(acc_start),
        .skip_load(1'b1),                     // the MFCC wrote FMEM directly
        .busy(act_busy), .done(act_done),
        // The stream-loader port is used by testbench
        .in_valid(1'b0), .in_data(8'd0), .in_last(1'b0), .in_ready(),
        .map_we(mfcc_we), .map_addr(mfcc_addr), .map_data(mfcc_data),
        .logits_o(logits), .class_o(class_o), .max_o(max_o), .margin_o(margin_o),
        .result_valid(), .cycles_o(cycles_o), .layer_o(layer_o));

    // ---- the one request that starts a run ---------------------------------
    wire run_req = key_press & (seq == S_KEY) & ~mfcc_busy & ~act_busy;

    // =========================================================================
    // sequencer
    //   S_INIT   wait for the codec's I2C init
    //   S_KEY    idle until a key press
    //   S_MFCC   mfcc_top runs on that snapshot (~21 ms)
    //   S_RUN    wait for the accelerator, then report
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seq        <= S_INIT;
            wm_go      <= 1'b0;
            wm_started <= 1'b0;
            wm_ok      <= 1'b0;
            wm_to      <= 24'hFF_FFFF;
            acc_start <= 1'b0;
            mfcc_start  <= 1'b0;
        end else begin
            // only the one-cycle pulses get per-cycle defaults; snap_* are state
            // and must hold their value (clearing them here stopped the copy
            // counter from ever advancing)
            acc_start  <= 1'b0;
            mfcc_start <= 1'b0;
            case (seq)
            // WM8960_init treats Go as a LEVEL with priority over the table counter
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
                    // Init failed, then perform failure indication
                    if (wm_to == 24'd0) seq <= S_KEY;
                end
            end
            // latch the oldest end of the ring, then copy 16000 samples into the
            // MFCC's buffer (it needs `start` first, so the pulse goes out here)
            S_KEY: if (run_req) begin
                mfcc_start <= 1'b1;           // 1-cycle pulse; mfcc latches its base
                seq        <= S_MFCC;
            end
            S_MFCC: if (mfcc_done) begin
                acc_start <= 1'b1;        // FMEM already holds the map
                seq       <= S_RUN;
            end
            S_RUN: if (act_done) seq <= S_KEY;
            default: seq <= S_INIT;
            endcase
        end
    end

    // ---- UART_TX FSM -------------------------------------------------------
    reg  [3:0]  class_q;
    reg         new_result;
    reg  [1:0]  tx_state;
    reg  [7:0]  tx_byte;
    reg         tx_send;
    wire        tx_busy;

    localparam [1:0] T_IDLE = 2'd0, T_CLASS = 2'd1, T_MARGIN = 2'd2, T_WAIT = 2'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state   <= T_IDLE;
            tx_byte    <= 8'd0;
            tx_send    <= 1'b0;
        end else begin
            tx_send <= 1'b0;
            case (tx_state)
            T_IDLE: if (act_done) begin
                tx_byte  <= 8'hA5;
                tx_send  <= 1'b1;
                tx_state <= T_CLASS;
            end
            T_CLASS: if (!tx_busy && !tx_send) begin
                tx_byte  <= {4'h0, class_o};
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
            if (act_done)          hold <= 23'h7F_FFFF;
            else if (hold != 23'd0) hold <= hold - 23'd1;
        end
    end

    assign led[0] = hb[24];                                   // ~1.4 Hz
    assign led[1] = !wm_ok            ? hb[21] :              // ~5.7 Hz: codec not configured
                    (mfcc_busy | act_busy) ? 1'b1  :          // computing
                    (hold != 23'd0)    ? 1'b1  :              // new result
                                         1'b0;

endmodule
