// =============================================================================
// tb_mel_fb.v -- FFT + mel filterbank, bit-exact against the integer model.
//
// This is the first *integration* test of the MFCC front end: the FFT streams its
// 256 power bins straight into the mel engine's power buffer, and the 40 mel
// energies are compared with the same integer arithmetic (unsigned Q1.15
// coefficients, >>15, 54-bit accumulator) that the RTL implements.
//
// Golden data comes from scripts/golden/gen_fft_pow_vectors.py, which drives the
// bit-level FFT model and then the integer mel model.
//
// Run from the project root:  bash sim/run.sh mel
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module tb_mel_fb;

    localparam integer N     = 512;
    localparam integer LOG2N = 9;
    localparam integer DW    = 24;
    localparam integer NBIN  = N / 2;
    localparam integer NFILT = `KWS_H_IN;   // mel bands = input height
    localparam integer MF_BITS = 54;
    localparam integer NVEC  = 6;
    localparam real    CLK_NS = 20.833;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    // ---- FFT ---------------------------------------------------------------
    reg  signed [DW-1:0] x_in = 0;
    reg                  x_valid = 1'b0;
    wire [LOG2N-2:0]     y_index;
    wire [2*DW:0]        y_pow;
    wire                 y_valid;
    wire                 fft_busy;
    wire [7:0]           drop_cnt;

    fft_core #(.N(N), .LOG2N(LOG2N), .DW(DW), .TW(16)) u_fft (
        .clk(clk), .rst_n(rst_n), .x_in(x_in), .x_valid(x_valid),
        .y_index(y_index), .y_pow(y_pow), .y_valid(y_valid),
        .busy(fft_busy), .drop_cnt(drop_cnt));

    // ---- mel ---------------------------------------------------------------
    reg                  mel_start = 1'b0;
    wire                 mel_busy, mel_done;
    wire                 mel_valid;
    wire [5:0]           mel_idx;
    wire [MF_BITS-1:0]   mel_data;
    reg  [NBIN-1:0]      bins_seen = 0;

    // the FFT's power stream is exactly the mel engine's write port
    mel_fb #(.NBIN(NBIN), .NFILT(NFILT), .MF_BITS(MF_BITS)) u_mel (
        .clk(clk), .rst_n(rst_n),
        .pw_we(y_valid), .pw_addr(y_index), .pw_data(y_pow[47:0]),
        .start(mel_start), .busy(mel_busy), .done(mel_done),
        .mel_valid(mel_valid), .mel_idx(mel_idx), .mel_data(mel_data));

    // ---- golden ------------------------------------------------------------
    reg [8*24-1:0]     names [0:NVEC-1];
    reg [DW-1:0]       x_mem [0:N-1];
    reg [MF_BITS-1:0]  exp_mel [0:NFILT-1];
    reg [MF_BITS-1:0]  got_mel [0:NFILT-1];
    reg [8*64-1:0]     path;

    integer errors = 0, i, v, nout, cycles;
    reg     counting;

    always @(posedge clk) begin
        if (rst_n && y_valid) begin
            bins_seen[y_index] <= 1'b1;
            if (&bins_seen && !counting) ;
        end
        if (rst_n && mel_valid) begin
            got_mel[mel_idx] <= mel_data;
            nout = nout + 1;
        end
        if (counting) cycles = cycles + 1;
    end

    task run_vector;
        input integer vi;
        begin
            $sformat(path, "golden/fft/in_%0s.mem", names[vi]);
            $readmemh(path, x_mem);
            $sformat(path, "golden/fft/expmel_%0s.mem", names[vi]);
            $readmemh(path, exp_mel);

            bins_seen = {NBIN{1'b0}};
            nout      = 0;
            mel_start = 1'b0;
            counting  = 1'b0;
            cycles    = 0;

            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                x_valid = 1'b1;
                x_in    = x_mem[i];
            end
            @(negedge clk);
            x_valid = 1'b0;

            // wait for the whole power spectrum, then kick the mel engine
            for (i = 0; i < 40000 && !(&bins_seen); i = i + 1) @(negedge clk);
            @(negedge clk);
            mel_start = 1'b1;
            @(negedge clk);
            mel_start = 1'b0;
            counting  = 1'b1;

            for (i = 0; i < 40000 && nout < NFILT; i = i + 1) @(negedge clk);
            counting = 1'b0;
            @(negedge clk);

            for (i = 0; i < NFILT; i = i + 1) begin
                if (got_mel[i] !== exp_mel[i]) begin
                    errors = errors + 1;
                    if (errors < 8)
                        $display("    FAIL %0s filter %0d: got %014x expected %014x",
                                 names[vi], i, got_mel[i], exp_mel[i]);
                end
            end
            $display("  %-9s %0d mel energies, %0d cycles", names[vi], nout, cycles);
            if ($test$plusargs("dump") && vi == 0) begin
                for (i = 0; i < 5; i = i + 1)
                    $display("    f%0d got %014x exp %014x", i, got_mel[i], exp_mel[i]);
                $display("    span[3] base/len/lo = %0d/%0d/%0d",
                         u_mel.span[24:15], u_mel.span[14:8], u_mel.span[7:0]);
                $display("    pw[6..8] = %012x %012x %012x",
                         u_mel.pw_mem[6], u_mel.pw_mem[7], u_mel.pw_mem[8]);
                $display("    coef[base..] = %04x %04x %04x",
                         u_mel.u_coef.dout, u_mel.coef_q, u_mel.pw_q[47:32]);
            end
        end
    endtask

    initial begin
        names[0] = "impulse";  names[1] = "sine8";    names[2] = "multitone";
        names[3] = "noise";    names[4] = "max_pos";  names[5] = "max_neg";
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_mel.vcd");
            $dumpvars(0, tb_mel_fb);
        end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        for (v = 0; v < NVEC; v = v + 1) run_vector(v);

        if (drop_cnt !== 8'd0) begin
            errors = errors + 1;
            $display("  FAIL: FFT input FIFO dropped %0d samples", drop_cnt);
        end
        if (errors == 0)
            $display("MEL_FB: %0d vectors x %0d filters bit-exact", NVEC, NFILT);
        else
            $display("MEL_FB: %0d mismatches", errors);
        $finish;
    end

endmodule
