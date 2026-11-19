// =============================================================================
// i2s_rx.v -- WM8960 ADC data in, 16-bit PCM samples out in the system domain.
//
// WM8960 is configured as I2S master: it drives BCLK and LRCLK and sends
// left-justified 16-bit signed samples on ADCDAT, MSB first, one BCLK after the
// LRCLK edge (the I2S "one bit delay").  At 16 kHz with 32-bit slots BCLK is
// 512 kHz, so a sample is stable on ADCDAT for ~2 us -- plenty for a 2-FF
// synchroniser, but the DATA still has to be captured with BCLK, so the
// receiving side runs in the BCLK domain and only the "sample ready" strobe
// crosses into the system domain.
//
//   BCLK domain:   shift ADCDAT into a 16-bit register on every BCLK rising edge;
//                  a LRCLK edge starts a new slot, and the previous slot's data
//                  is latched with a toggling "ready" flag.
//   system domain: sync the toggle through two FFs, and on the transition
//                  present the (now stable) sample as a one-cycle valid pulse.
//
// Latency: one slot (62.5 us at 16 kHz) plus 2-3 system clocks.  Irrelevant
// next to the 10 ms hop.
//
// Only the LEFT channel is used; the WM8960 is configured mono-ish by feeding
// the same ADC to both slots.
// =============================================================================
`timescale 1ns / 1ps

module i2s_rx #(
    parameter integer DW = 16
) (
    input  wire          clk,        // system clock (48 MHz)
    input  wire          rst_n,

    // ---- I2S pins from the WM8960 (BCLK domain) ----------------------------
    input  wire          bclk,
    input  wire          lrclk,
    input  wire          adc_dat,

    // ---- samples in the system domain --------------------------------------
    output reg           s_valid,    // one clk pulse per sample
    output reg  signed [DW-1:0] s_data
);

    // =========================================================================
    // BCLK domain
    // =========================================================================
    // reset for the BCLK domain: assert asynchronously, release synchronously
    reg [1:0] brst_sync;
    always @(posedge bclk or negedge rst_n)
        if (!rst_n) brst_sync <= 2'b00;
        else        brst_sync <= {brst_sync[0], 1'b1};
    wire brst_n = brst_sync[1];

    reg [15:0] shreg;          // holds the current slot, MSB first
    reg [4:0]  bitcnt;         // 0..15 within a slot
    reg        lr_q;           // previous LRCLK, for edge detection
    reg        left_q;         // which channel the finished slot belonged to
    reg [15:0] slot;           // completed slot
    reg        ready_tog;      // toggles once per completed left slot

    always @(posedge bclk or negedge brst_n) begin
        if (!brst_n) begin
            shreg     <= 16'd0;
            bitcnt    <= 5'd0;
            lr_q      <= 1'b0;
            left_q    <= 1'b0;
            slot      <= 16'd0;
            ready_tog <= 1'b0;
        end else begin
            lr_q <= lrclk;

            // LRCLK edge: close the slot that just finished, open the next one
            if (lrclk != lr_q) begin
                if (left_q) begin
                    slot      <= shreg;      // keep the left channel
                    ready_tog <= ~ready_tog;
                end
                left_q <= ~lrclk;            // low = left channel in I2S
                bitcnt <= 5'd0;
                // I2S sends the MSB one BCLK AFTER the LRCLK edge, so do NOT
                // shift here -- shifting at the edge costs one bit and every
                // sample comes out exactly doubled.
                shreg  <= shreg;
            end else if (bitcnt != 5'd16) begin
                shreg  <= {shreg[14:0], adc_dat};
                bitcnt <= bitcnt + 5'd1;
            end
        end
    end

    // =========================================================================
    // system domain: strobe synchroniser
    // =========================================================================
    reg [2:0] tog_sync;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) tog_sync <= 3'b000;
        else        tog_sync <= {tog_sync[1:0], ready_tog};

    wire new_sample = tog_sync[2] ^ tog_sync[1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_valid <= 1'b0;
            s_data  <= {DW{1'b0}};
        end else begin
            s_valid <= new_sample;
            if (new_sample) s_data <= slot[15:0];
        end
    end

endmodule
