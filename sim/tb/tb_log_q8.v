// =============================================================================
// tb_log_q8.v -- log unit test: 40 mel energies per vector -> 40 Q8.8 logs.
//
// Golden data: scripts/golden/gen_log_lut.py evaluates the *same* integer recipe
// the RTL implements (exponent + 8-bit mantissa table + constant + rounding), on
// the mel values that scripts/golden/gen_fft_pow_vectors.py produced from the
// bit-exact FFT + mel models.  So the three blocks are chained end to end here:
// FFT -> mel -> log, each verified against its own bit-level model.
//
// Run from the project root:  bash sim/run.sh log
// =============================================================================
`timescale 1ns / 1ps

module tb_log_q8;

    localparam integer MEL_BITS = 54;
    localparam integer NFILT    = 40;
    localparam integer NVEC     = 6;
    localparam real    CLK_NS   = 20.833;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    reg                    in_valid = 1'b0;
    reg  [MEL_BITS-1:0]    mel_i = 0;
    wire                   out_valid;
    wire signed [15:0]     log_o;

    log_q8 #(.MEL_BITS(MEL_BITS)) u_log (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .mel_i(mel_i),
        .out_valid(out_valid), .log_o(log_o));

    reg [8*24-1:0]    names [0:NVEC-1];
    reg [MEL_BITS-1:0] mel_mem [0:NFILT-1];
    reg [15:0]         exp_mem [0:NFILT-1];
    reg [15:0]         got_mem [0:NFILT-1];
    reg [8*64-1:0]     path;
    integer errors = 0, i, v, nout;

    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            got_mem[nout] <= log_o;
            nout = nout + 1;
        end
    end

    task run_vector;
        input integer vi;
        begin
            $sformat(path, "golden/fft/expmel_%0s.mem", names[vi]);
            $readmemh(path, mel_mem);
            $sformat(path, "golden/fft/explog_%0s.mem", names[vi]);
            $readmemh(path, exp_mem);
            nout = 0;
            if ($test$plusargs("dump"))
                $display("    mel[0..2] = %014x %014x %014x", mel_mem[0], mel_mem[1], mel_mem[2]);
            for (i = 0; i < NFILT; i = i + 1) begin
                @(negedge clk);
                in_valid = 1'b1;
                mel_i    = mel_mem[i];
                @(negedge clk);
                in_valid = 1'b0;
                mel_i    = 0;
            end
            for (i = 0; i < 20; i = i + 1) @(negedge clk);
            for (i = 0; i < NFILT; i = i + 1)
                if (got_mem[i] !== exp_mem[i]) begin
                    errors = errors + 1;
                    if (errors < 8)
                        $display("    FAIL %0s filter %0d: got %04x expected %04x",
                                 names[vi], i, got_mem[i], exp_mem[i]);
                end
            $display("  %-9s %0d logs, first %0d", names[vi], nout, $signed(got_mem[0]));
        end
    endtask

    initial begin
        names[0] = "impulse";  names[1] = "sine8";    names[2] = "multitone";
        names[3] = "noise";    names[4] = "max_pos";  names[5] = "max_neg";
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_log.vcd");
            $dumpvars(0, tb_log_q8);
        end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        for (v = 0; v < NVEC; v = v + 1) run_vector(v);
        if (errors == 0)
            $display("LOG_Q8: %0d vectors x %0d logs bit-exact", NVEC, NFILT);
        else
            $display("LOG_Q8: %0d mismatches", errors);
        $finish;
    end

endmodule
