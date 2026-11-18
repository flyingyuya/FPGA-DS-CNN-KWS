// =============================================================================
// tb_dct40.v -- 40x40 DCT-II unit test, bit-exact against the integer model.
//
// Feeds the log-mel vectors produced by gen_log_lut.py and compares the 40 DCT
// coefficients with gen_dct_rom.py's model, which uses the same coefficient ROM
// and the same rounding (accumulate, then one rounded shift by 15).
//
// Run from the project root:  bash sim/run.sh dct
// =============================================================================
`timescale 1ns / 1ps

module tb_dct40;

    localparam integer N     = 40;
    localparam integer NVEC  = 6;
    localparam integer DBITS = 24;
    localparam real    CLK_NS = 20.833;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    reg               log_we = 1'b0;
    reg  [5:0]        log_addr = 6'd0;
    reg  signed [15:0] log_data = 16'sd0;
    reg               start = 1'b0;
    wire              busy, done;
    wire              dct_valid;
    wire [5:0]        dct_idx;
    wire signed [DBITS-1:0] dct_data;

    dct40 #(.N(N), .DCT_BITS(DBITS)) u_dct (
        .clk(clk), .rst_n(rst_n),
        .log_we(log_we), .log_addr(log_addr), .log_data(log_data),
        .start(start), .busy(busy), .done(done),
        .dct_valid(dct_valid), .dct_idx(dct_idx), .dct_data(dct_data));

    reg [8*24-1:0]  names [0:NVEC-1];
    reg signed [15:0] log_mem [0:N-1];
    reg signed [DBITS-1:0] exp_mem [0:N-1];
    reg signed [DBITS-1:0] got_mem [0:N-1];
    reg [8*64-1:0]  path;
    integer errors = 0, i, v, nout, cycles;
    reg running;

    always @(posedge clk) begin
        if (rst_n && dct_valid) begin
            got_mem[dct_idx] <= dct_data;
            nout = nout + 1;
        end
        if (running) cycles = cycles + 1;
    end

    task run_vector;
        input integer vi;
        begin
            $sformat(path, "golden/fft/explog_%0s.mem", names[vi]);
            $readmemh(path, log_mem);
            $sformat(path, "golden/fft/expdct_%0s.mem", names[vi]);
            $readmemh(path, exp_mem);

            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                log_we   = 1'b1;
                log_addr = i[5:0];
                log_data = log_mem[i];
            end
            @(negedge clk);
            log_we = 1'b0;

            nout    = 0;
            cycles  = 0;
            start   = 1'b1;
            running = 1'b1;
            @(negedge clk);
            start = 1'b0;
            for (i = 0; i < 8000 && nout < N; i = i + 1) @(negedge clk);
            running = 1'b0;
            @(negedge clk);

            for (i = 0; i < N; i = i + 1)
                if (got_mem[i] !== exp_mem[i]) begin
                    errors = errors + 1;
                    if (errors < 8)
                        $display("    FAIL %0s c%0d: got %0d expected %0d",
                                 names[vi], i, got_mem[i], exp_mem[i]);
                end
            $display("  %-9s %0d coefficients, %0d cycles", names[vi], nout, cycles);
            if ($test$plusargs("dump") && vi == 0) begin
                $display("    logbuf[0..3] = %0d %0d %0d %0d", u_dct.logbuf[0], u_dct.logbuf[1], u_dct.logbuf[2], u_dct.logbuf[3]);
                $display("    coef_mem[0..3] = %0d %0d %0d %0d", u_dct.coef_mem[0], u_dct.coef_mem[1], u_dct.coef_mem[2], u_dct.coef_mem[3]);
            end
        end
    endtask

    initial begin
        names[0] = "impulse";  names[1] = "sine8";    names[2] = "multitone";
        names[3] = "noise";    names[4] = "max_pos";  names[5] = "max_neg";
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_dct.vcd");
            $dumpvars(0, tb_dct40);
        end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        for (v = 0; v < NVEC; v = v + 1) run_vector(v);
        if (errors == 0)
            $display("DCT40: %0d vectors x %0d coefficients bit-exact", NVEC, N);
        else
            $display("DCT40: %0d mismatches", errors);
        $finish;
    end

endmodule
