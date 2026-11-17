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
// M6 replaces the fixed frame with the WM8960 + on-chip MFCC path and streams the
// frame through `in_valid/in_data/in_last` (the port already exists).
//
// The key is synchronised and edge detected; a mechanical press is several
// milliseconds, longer than one inference, so the edge detector alone is enough
// and no debounce counter is needed.
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module board_top #(
    parameter integer NUM_CLASSES = `KWS_NUM_CLASSES,
    parameter         MEM_DIR     = "mem"
) (
    input  wire sys_clk_p,
    input  wire sys_clk_n,
    input  wire sys_rstn,
    input  wire key,             // W21
    output wire uart_tx,         // P15
    output wire [1:0] led        // W22 / Y22
);

    // ---- clocking ----------------------------------------------------------
    wire clk, rst_n, locked;
    clk_gen u_clk (
        .clk_p(sys_clk_p), .clk_n(sys_clk_n),
        .rst_async_n(sys_rstn),
        .clk(clk), .rst_n(rst_n), .locked(locked)
    );

    // ---- key: synchronise and detect a press --------------------------------
    reg [2:0] ks;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) ks <= 3'b000;
        else        ks <= {ks[1:0], key};

    wire key_press = (ks[2:1] == 2'b01);      // rising edge of the (active-high) key

    // ---- accelerator -------------------------------------------------------
    wire        busy, done, result_valid;
    wire [NUM_CLASSES*16-1:0] logits;
    wire [3:0]  class_o;
    wire signed [15:0] max_o, margin_o;
    wire [31:0] cycles_o;
    wire [3:0]  layer_o;

    accelerator_top #(.NUM_CLASSES(NUM_CLASSES), .MEM_DIR(MEM_DIR)) u_accel (
        .clk(clk), .rst_n(rst_n),
        .start(key_press),
        .skip_load(1'b1),                     // frame is preloaded in FMEM
        .busy(busy), .done(done),
        .in_valid(1'b0), .in_data(8'd0), .in_last(1'b0), .in_ready(),
        .logits_o(logits), .class_o(class_o), .max_o(max_o), .margin_o(margin_o),
        .result_valid(result_valid), .cycles_o(cycles_o), .layer_o(layer_o));

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

    assign led = {new_result, busy};

endmodule
