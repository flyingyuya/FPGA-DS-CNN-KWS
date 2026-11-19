// =============================================================================
// tb_i2s_rx.v -- drive a WM8960-style I2S stream and check every sample.
//
// The transmitter below is written the way the codec works: it changes ADCDAT on
// the falling edge of BCLK and both sides sample it on the rising edge, with the
// MSB appearing one BCLK after the LRCLK edge.  A sample is only correct if the
// receiver tolerates that one-bit delay -- an off-by-one bit alignment shows up
// as "every sample is exactly 2x the expected value".
//
// Run from the project root:  bash sim/run.sh i2s
// =============================================================================
`timescale 1ns / 1ps

module tb_i2s_rx;

    localparam integer NS  = 64;          // samples to push
    localparam real    BCLK_HALF = 976.5625; // 512 kHz -> 1.953 us period
    localparam real    CLK_NS    = 20.833;   // 48 MHz

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    wire s_valid;
    wire signed [15:0] s_data;

    reg bclk    = 1'b0;
    reg lrclk   = 1'b1;   // idle high (right slot)
    reg adc_dat = 1'b0;

    always #(BCLK_HALF) bclk = ~bclk;

    i2s_rx u_dut (.clk(clk), .rst_n(rst_n), .bclk(bclk), .lrclk(lrclk),
                  .adc_dat(adc_dat), .s_valid(s_valid), .s_data(s_data));

    reg signed [15:0] stim [0:NS-1];
    integer got = 0, errors = 0, k;

    // the receiver must present one sample per completed left slot
    always @(posedge clk) begin
        if (rst_n && s_valid) begin
            if (got >= NS) begin
                $display("  extra sample %0d", s_data);
                errors = errors + 1;
            end else if (s_data !== stim[got]) begin
                $display("  sample %0d: got %0d, expected %0d", got, s_data, stim[got]);
                errors = errors + 1;
            end
            got = got + 1;
        end
    end

    task send_slot;
        input integer chi;             // 0 = left
        input signed [15:0] v;
        integer b;
        begin
            @(negedge bclk);
            lrclk = chi[0];            // left = low
            adc_dat = 1'b0;            // I2S one-bit delay
            for (b = 0; b < 16; b = b + 1) begin
                @(negedge bclk);
                adc_dat = v[15 - b];
            end
            @(negedge bclk);
            adc_dat = 1'b0;            // rest of the 32-bit slot is padding
        end
    endtask

    initial begin
        for (k = 0; k < NS; k = k + 1)
            stim[k] = (k % 2) ? -(16'sd1000 + k) : (16'sd1000 + k * 7);
        stim[0] = 16'sh1234;           // MSB-heavy patterns to catch bit slips
        stim[1] = -16'sh1234;
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_i2s.vcd");
            $dumpvars(0, tb_i2s_rx);
        end
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        for (k = 0; k < NS; k = k + 1) begin
            send_slot(1, 16'sd7);      // right slot: must be ignored
            send_slot(0, stim[k]);
        end
        send_slot(1, 16'sd0);          // a slot must *close* the last left sample
        repeat (40) @(negedge clk);

        if (errors == 0) $display("I2S_RX: %0d/%0d samples bit-exact, right slot ignored", got, NS);
        else             $display("I2S_RX: %0d errors over %0d samples", errors, NS);
        $finish;
    end

endmodule
