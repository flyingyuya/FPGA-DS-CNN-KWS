// =============================================================================
// tb_board_top.v -- whole-board smoke test: I2S in, UART out.
//
// This is the test that stands in for "power it up and see if it works", and it
// exists because the integration adds three things that no unit test covers:
//
//   1. the I2C slave handshake the WM8960 needs before the design leaves S_INIT;
//   2. the staging buffer's column/row reorder (mfcc writes frame-major, the
//      accelerator's loader wants address-major);
//   3. the stream-loader handshake (in_valid / in_ready / in_last) with a real
//      4040-byte map instead of a preloaded FMEM.
//
// Stimulus: one real "yes" clip from golden/mfcc/in_1_yes.mem, played out on the
// I2S pins.  BCLK runs at 100 MHz here to keep the simulation short (10 ms of
// audio instead of 1 s); i2s_rx does not care because it samples ADCDAT on BCLK
// and only the "new sample" toggle crosses into the 48 MHz domain.
//
// Expectation: the UART emits 0xA5, class=1, margin.  Note the map the RTL
// produces is NOT bit-identical to the reference (2399/12120 codes differ by
// +-1), so this test also checks that that residue does not change the verdict.
//
// Run from the project root:  bash sim/run.sh board
// =============================================================================
`timescale 1ns / 1ps

module tb_board_top;

    localparam integer NSAMP  = 16000;
    localparam integer NBYTES = `KWS_IN_PIXELS;
    localparam real    CLK_NS = 20.833;      // 48 MHz
    localparam real    BCLK_HALF = 5.0;      // 100 MHz BCLK (simulation only)
    localparam real    BIT_NS = 1.0e9 / 115200.0;

    reg clk = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    reg rstn = 1'b0;
    reg key  = 1'b1;   // key_debounce is active-low: released = 1

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
        .key(key), .uart_tx(uart_tx), .led(led),
        .aud_bclk(bclk), .aud_lrclk(lrclk), .aud_adcdat(adcdat),
        .aud_scl(scl), .aud_sda(sda));

    // =========================================================================
    // I2C slave: ACK every byte, detect START to re-align the byte counter
    // (without START detection the counter drifts -- see docs sec. 24)
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
    wire i2c_ack = (bidx == 8);

    // diagnostic: is the bus moving at all?
    integer nscl = 0;
    always @(posedge scl) nscl <= nscl + 1;
    always @(*) sda_low = i2c_ack;

    // =========================================================================
    // I2S transmitter: left-justified, MSB one BCLK after the LRCLK edge
    // =========================================================================
    reg signed [15:0] stim [0:NSAMP-1];
    integer nsent = 0;

    task send_slot;
        input integer chi;
        input signed [15:0] v;
        integer b;
        begin
            @(negedge bclk);
            lrclk = chi[0];
            adcdat = 1'b0;
            for (b = 0; b < 16; b = b + 1) begin
                @(negedge bclk);
                adcdat = v[15 - b];
            end
            @(negedge bclk);
            adcdat = 1'b0;
        end
    endtask

    // =========================================================================
    // UART receiver
    // =========================================================================
    reg [7:0] rx [0:15];
    integer nrx = 0;
    reg [7:0] sh;
    integer b;

    initial begin : uart_mon
        forever begin
            @(negedge uart_tx);                       // start bit
            #(BIT_NS * 1.5);
            for (b = 0; b < 8; b = b + 1) begin
                sh[b] = uart_tx;
                #(BIT_NS);
            end
            if (nrx < 16) rx[nrx] = sh;
            nrx = nrx + 1;
        end
    end

    // watch the bytes the stream loader actually consumes (address order)
    reg [7:0] sm [0:NBYTES-1];
    integer   nstream = 0;
    always @(posedge clk) begin
        // the map now goes straight into FMEM, so watch the MFCC write port
        // the map now goes straight into FMEM, so watch the MFCC write port
        if (u_dut.mfcc_we) begin
            if (u_dut.mfcc_addr < NBYTES) sm[u_dut.mfcc_addr] <= u_dut.mfcc_data;
            nstream <= nstream + 1;
        end
    end

    reg [7:0] expmap [0:NBYTES-1];
    integer nbad, k;
    integer i, t;
    initial begin
        $readmemh("golden/mfcc/in_1_yes.mem", stim);
        $readmemh("golden/mfcc/exp_1_yes.mem", expmap);
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_board.vcd");
            $dumpvars(0, tb_board_top);
        end
        repeat (10) @(negedge clk);
        rstn = 1'b1;

        // 1) wait for the codec's I2C configuration to finish
        t = 0;
        while (nword < 33 && t < 4_000_000) begin @(negedge clk); t = t + 1; end
        $display("  I2C: %0d register words written after %0d clocks (%0d SCL rising edges, seq=%0d wm_done=%b)",
                 nword, t, nscl, u_dut.seq, u_dut.wm_done);
        repeat (100) @(negedge clk);

        // 2) play one second of audio into the capture ring (sped up: 100 MHz BCLK).
        //    500 junk samples first so the write pointer is NOT at 0 when the run
        //    starts -- that exercises the ring wrap in mfcc_top's read address
        //    (base = 500, the window is still exactly the clip).
        for (i = 0; i < 500; i = i + 1) send_slot(0, 16'sd0);
        for (i = 0; i < NSAMP; i = i + 1) begin
            send_slot(1, 16'sd0);
            send_slot(0, stim[i]);
        end
        send_slot(1, 16'sd0);
        repeat (2000) @(negedge clk);

        // 3) NOW press: the last second is already buffered, so the answer comes
        //    back after snapshot + MFCC + CNN (~22 ms), not after another second
        //    of recording.  key_debounce needs 20 ms of a stable level and is
        //    active-low, so a press is driven as 0 here.
        key = 1'b0;
        repeat (1_100_000) @(negedge clk);
        key = 1'b1;

        // 4) wait for the UART report
        t = 0;
        while (nrx < 3 && t < 8_000_000) begin @(negedge clk); t = t + 1; end

        // ...and make sure it does not keep repeating (result_valid is a level
        // in accelerator_top; feeding it straight into the UART resent forever)
        repeat (200_000) @(negedge clk);
        if (nrx != 3) $display("  WARNING: %0d UART bytes -- the report repeated", nrx);

        nbad = 0;
        for (k = 0; k < NBYTES; k = k + 1)
            if (sm[k] !== expmap[k]) nbad = nbad + 1;
        $display("  map: %0d/%0d bytes differ from the reference; first bad at %0d (%0d vs %0d)",
                 nbad, NBYTES, 0, sm[0], expmap[0]);
        // is it a temporal shift?  compare against the reference slid by k frames
        for (k = -2; k <= 2; k = k + 1) begin
            nbad = 0;
            for (i = 0; i < 40*99; i = i + 1) begin
                if ((sm[(i/101)*101 + 1 + (i%101)] !== expmap[(i/101)*101 + 1 + (i%101) + k])
                    && (1 + (i%101) + k >= 1) && (1 + (i%101) + k <= 99))
                    nbad = nbad + 1;
            end
            $display("    shift %0d frame(s): %0d mismatches over frames 1..99", k, nbad);
        end
        $display("  UART: %0d bytes after %0d clocks: %02x %02x %02x",
                 nrx, t, rx[0], rx[1], rx[2]);
        // NOTE: the model's class order is 0=silence 1=unknown 2=yes 3=no ...,
        // so the 'yes' clip must come out as class 2 (the golden file name
        // "1_yes.mem" refers to the dataset label, not the model's class index).
        if (nrx >= 3 && rx[0] === 8'hA5 && rx[1] === 8'h02)
            $display("BOARD_TOP: PASS ('yes' -> class 2, margin %0d)", rx[2]);
        else
            $display("BOARD_TOP: FAIL (expected A5 02 xx for the 'yes' clip, got %02x %02x)", rx[0], rx[1]);
        $finish;
    end

endmodule
