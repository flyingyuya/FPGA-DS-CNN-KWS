// =============================================================================
// tb_board_top.v -- whole-board smoke test: I2S in, UART out.
//
// Stands in for "power it up and see if it works".  It covers the three things
// no unit test can:
//   1. the I2C handshake the WM8960 needs before the design leaves S_INIT;
//   2. the streaming front end + FMEM direct write + accelerator chain;
//   3. the 5-byte UART report including its checksum.
//
// Stimulus: one real "yes" clip from golden/mfcc/in_1_yes.mem played on the I2S
// pins AFTER the key press (streaming captures from the press onwards).  BCLK
// runs at 100 MHz to keep the simulation short, so the source pauses once per
// 160 samples to stay inside the front end's ring slack -- at the real 16 kHz it
// never needs to.
//
// Expectation: A5 02 <margin_hi> <margin_lo> <checksum>, i.e. class 2.
// (The model's class order is 0=silence 1=unknown 2=yes ..., while the golden
//  file name "1_yes.mem" refers to the dataset label.  Easy to get wrong.)
//
// Run from the project root:  bash sim/run.sh board
// =============================================================================
`timescale 1ns / 1ps

module tb_board_top;

    localparam integer NSAMP  = 16000;
    localparam real    CLK_NS = 20.833;      // 48 MHz
    localparam real    BCLK_HALF = 5.0;      // 100 MHz BCLK (simulation only)
    localparam integer BIT_CLKS = (48_000_000 + 115200/2) / 115200;

    reg clk = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    reg rstn = 1'b0;
    reg key  = 1'b1;   // key_debounce is active-low: released = 1
    reg key2 = 1'b1;   // mode toggle, held released -> single-shot

    wire uart_tx;
    wire [1:0] led;

    // ---- I2C open-drain bus ------------------------------------------------
    wire scl, sda;
    reg  sda_low = 1'b0;
    assign sda = sda_low ? 1'b0 : 1'bz;
    wire sda_in = (sda === 1'b0) ? 1'b0 : 1'b1;

    // ---- I2S ---------------------------------------------------------------
    reg bclk = 1'b0, lrclk = 1'b1, adcdat = 1'b0;
    always #(BCLK_HALF) bclk = ~bclk;

    board_top #(.WM_DLY_MS(0)) u_dut (
        .sys_clk_p(clk), .sys_clk_n(1'b0), .sys_rstn(rstn),
        .key1(key), .key2(key2), .uart_tx(uart_tx), .led(led),
        .aud_bclk(bclk), .aud_lrclk(lrclk), .aud_adcdat(adcdat),
        .aud_scl(scl), .aud_sda(sda));

    // =========================================================================
    // I2C slave: ACK every byte, detect START so the byte counter stays aligned
    // =========================================================================
    reg [7:0] cur = 8'd0;
    integer bidx = 0, bcnt = 0, nword = 0;
    reg tr = 1'b0;

    always @(negedge sda) if (scl) tr <= 1'b1;

    always @(posedge scl) if (rstn) begin
        if (tr) begin
            tr <= 1'b0; bidx <= 0; bcnt <= 0; cur <= 8'd0;
        end else if (bidx == 8) begin
            bidx <= 0;
            bcnt <= (bcnt == 2) ? 0 : bcnt + 1;
            if (bcnt == 2) nword <= nword + 1;
        end else begin
            cur  <= {cur[6:0], sda_in};
            bidx <= bidx + 1;
        end
    end
    always @(*) sda_low = (bidx == 8);       // ACK the 9th clock

    // =========================================================================
    // I2S transmitter: MSB one BCLK after the LRCLK edge
    // =========================================================================
    reg signed [15:0] stim [0:NSAMP-1];
    integer nsent = 0, k;

    task send_slot;
        input integer chi;
        input signed [15:0] v;
        integer sb;
        begin
            @(negedge bclk);
            lrclk  = chi[0];
            adcdat = 1'b0;
            for (sb = 0; sb < 16; sb = sb + 1) begin
                @(negedge bclk);
                adcdat = v[15 - sb];
            end
            @(negedge bclk);
            adcdat = 1'b0;
        end
    endtask

    // =========================================================================
    // Oversampling UART receiver (fixed #delays drift against the integer clock
    // divider of uart_tx)
    // =========================================================================
    reg [7:0] rx [0:15];
    integer   nrx = 0;
    reg [7:0] sh;
    integer   ub;

    initial begin : uart_mon
        forever begin
            @(negedge uart_tx);
            repeat (BIT_CLKS + BIT_CLKS/2) @(posedge clk);   // middle of bit 0
            for (ub = 0; ub < 8; ub = ub + 1) begin
                sh[ub] = uart_tx;
                repeat (BIT_CLKS) @(posedge clk);
            end
            if (nrx < 16) rx[nrx] = sh;
            nrx = nrx + 1;
        end
    end

    integer i, t;
    initial begin
        $readmemh("golden/mfcc/in_1_yes.mem", stim);
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_board.vcd");
            $dumpvars(0, tb_board_top);
        end
        repeat (10) @(negedge clk);
        rstn = 1'b1;

        // 1) wait for the codec's I2C configuration
        t = 0;
        while (nword < 33 && t < 4_000_000) begin @(negedge clk); t = t + 1; end
        $display("  I2C: %0d register words after %0d clocks", nword, t);
        repeat (100) @(negedge clk);

        // 2) press: streaming capture starts here, so the audio follows
        key = 1'b0;
        repeat (1_100_000) @(negedge clk);      // > 20 ms debounce
        key = 1'b1;

        // 3) play 1 s of audio, pausing once per 10 ms frame so the 100 MHz BCLK
        //    stays inside the front end's ring slack (real I2S is 16 kHz)
        for (i = 0; i < NSAMP; i = i + 1) begin
            send_slot(1, 16'sd0);
            send_slot(0, stim[i]);
            if (i % 160 == 159) repeat (8400) @(negedge clk);
        end
        send_slot(1, 16'sd0);

        // 4) 5-byte report: A5, class, margin_hi, margin_lo, checksum
        t = 0;
        while (nrx < 5 && t < 8_000_000) begin @(negedge clk); t = t + 1; end
        repeat (200_000) @(negedge clk);
        if (nrx != 5) $display("  WARNING: %0d UART bytes -- the report repeated", nrx);

        $display("  UART: %0d bytes after %0d clocks: %02x %02x %02x %02x %02x",
                 nrx, t, rx[0], rx[1], rx[2], rx[3], rx[4]);
        if (nrx >= 5 && (rx[0] ^ rx[1] ^ rx[2] ^ rx[3]) !== rx[4])
            $display("  WARNING: checksum mismatch");
        if (nrx >= 5 && rx[0] === 8'hA5 && rx[1] === 8'h02 &&
            (rx[0] ^ rx[1] ^ rx[2] ^ rx[3]) === rx[4])
            $display("BOARD_TOP: PASS ('yes' -> class 2, margin %0d)",
                     $signed({rx[2], rx[3]}));
        else
            $display("BOARD_TOP: FAIL (expected A5 02 <hi> <lo> <csum>)");

        // 5) KEY2 -> continuous mode: the next window must restart by itself
        fork
            begin
                key2 = 1'b0;
                repeat (1_100_000) @(negedge clk);
                key2 = 1'b1;
            end
            begin : wait_auto
                integer t2;
                reg seen;
                t2 = 0;
                seen = 1'b0;
                while (!seen && t2 < 3_000_000) begin
                    @(posedge clk);
                    t2 = t2 + 1;
                    if (u_dut.mfcc_start) seen = 1'b1;
                end
                if (seen) $display("  continuous mode: next capture restarted without a key press");
                else      $display("  continuous mode: FAIL (no automatic restart)");
            end
        join
        $finish;
    end

endmodule
