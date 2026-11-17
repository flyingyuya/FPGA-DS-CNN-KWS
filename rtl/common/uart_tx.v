// =============================================================================
// uart_tx.v -- minimal 8N1 transmitter, for reporting results to a PC.
//
// 115200 baud from 48 MHz: divider = 48e6 / 115200 = 416.67, rounded to 417,
// which is +0.08 % -- far inside the +-2 % a UART tolerates.
//
// One byte at a time, `busy` high while shifting.  The accelerator produces one
// classification every few milliseconds, so a single-byte buffer is plenty.
// =============================================================================
`timescale 1ns / 1ps

module uart_tx #(
    parameter integer CLK_HZ = 48_000_000,
    parameter integer BAUD   = 115_200
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       send,        // pulse: start sending tx_data
    input  wire [7:0] tx_data,
    output reg        tx,          // idle high
    output reg        busy
);

    localparam integer DIV = (CLK_HZ + BAUD/2) / BAUD;

    reg [15:0] cnt;
    reg [3:0]  bit_i;
    reg [9:0]  shifter;            // {stop, data[7:0], start}

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx      <= 1'b1;
            busy    <= 1'b0;
            cnt     <= 16'd0;
            bit_i   <= 4'd0;
            shifter <= 10'h3FF;
        end else if (!busy) begin
            tx <= 1'b1;
            if (send) begin
                shifter <= {1'b1, tx_data, 1'b0};
                busy    <= 1'b1;
                cnt     <= 16'd0;
                bit_i   <= 4'd0;
                tx      <= 1'b0;                 // start bit begins immediately
            end
        end else begin
            cnt <= cnt + 1'b1;
            if (cnt == DIV - 1) begin
                cnt <= 16'd0;
                if (bit_i == 4'd9) begin
                    busy <= 1'b0;
                    tx   <= 1'b1;
                end else begin
                    bit_i   <= bit_i + 1'b1;
                    shifter <= {1'b1, shifter[9:1]};
                    tx      <= shifter[1];
                end
            end
        end
    end

endmodule
