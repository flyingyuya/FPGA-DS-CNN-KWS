// =============================================================================
// tb_mfcc_top.v -- end-to-end MFCC front end.
//
// Feeds one second of 16 kHz PCM (preloaded into the sample buffer) and compares
// the 40 x 101 UINT8 map with the composed integer reference model
// (scripts/golden/mfcc_ref.py).  Mismatches are reported as a histogram by
// absolute difference, because "everything is off by exactly 1" and "some codes
// are wildly wrong" are very different failures.
//
// +dump also snapshots the mel engine's power buffer right after frame 0's mel
// completes, so the first stage that differs can be identified from the waveform
// without another round of probing.
//
// Run from the project root:  bash sim/run.sh mfcc [+dump]
// =============================================================================
`timescale 1ns / 1ps

module tb_mfcc_top;

    localparam integer NSAMP  = 16000;
    localparam integer NFRAME = 101;
    localparam integer NCODE  = 4040;
    localparam integer NVEC   = 3;
    localparam real    CLK_NS = 20.833;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    reg         start = 1'b0;
    wire        busy, done;
    wire        mfcc_we;
    wire [11:0] mfcc_addr;
    wire [7:0]  mfcc_data;
    wire [31:0] cycles_o;
    wire [7:0]  frame_o;
    wire        dbg_taken, dbg_taken2;
    wire [47:0] dbg_pw0;
    wire [53:0] dbg_mel0;
    wire [15:0] dbg_log0;
    wire [23:0] dbg_dct0;

    mfcc_top #(.NSAMP(NSAMP), .NFRAME(NFRAME), .SKIP_LOAD(1)) u_mfcc (
        .clk(clk), .rst_n(rst_n),
        .s_valid(1'b0), .s_data(16'sd0), .s_ready(),
        .start(start), .busy(busy), .done(done),
        .mfcc_we(mfcc_we), .mfcc_addr(mfcc_addr), .mfcc_data(mfcc_data),
        .cycles_o(cycles_o), .frame_o(frame_o),
        .dbg_taken(dbg_taken), .dbg_taken2(dbg_taken2), .dbg_pw0(dbg_pw0), .dbg_mel0(dbg_mel0), .dbg_log0(dbg_log0), .dbg_dct0(dbg_dct0));

    reg [8*24-1:0] names [0:NVEC-1];
    reg [7:0]      exp_mem [0:NCODE-1];
    reg [7:0]      got_mem [0:NCODE-1];
    reg [8*64-1:0] path;

    integer errors = 0, i, v, nwrite, dd, d1, dbig, dmax;
    integer errors_total = 0, firstbad, f0bad, fbad;

    // (a hierarchical probe of the mel engine's power buffer reads X under
    //  iverilog even though the engine consumes it -- the useful diagnostic is the
    //  |d| histogram below, plus comparing intermediate stage dumps from
    //  scripts/golden/mfcc_ref.py)

    always @(posedge clk) begin
        if (rst_n && mfcc_we) begin
            got_mem[mfcc_addr] <= mfcc_data;
            nwrite = nwrite + 1;
        end
    end

    task run_vector;
        input integer vi;
        begin
            $sformat(path, "golden/mfcc/in_%0s.mem", names[vi]);
            $readmemh(path, u_mfcc.samp);
            $sformat(path, "golden/mfcc/exp_%0s.mem", names[vi]);
            $readmemh(path, exp_mem);
            for (i = 0; i < NCODE; i = i + 1) got_mem[i] = 8'hxx;
            nwrite = 0;

            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            while (!done) @(negedge clk);
            @(negedge clk);

            errors = 0; d1 = 0; dbig = 0; dmax = 0; firstbad = -1; f0bad = 0; fbad = 0;
            for (i = 0; i < NCODE; i = i + 1)
                if (got_mem[i] !== exp_mem[i]) begin
                    errors = errors + 1;
                    dd = got_mem[i] - exp_mem[i];
                    if (dd < 0) dd = -dd;
                    if (dd > dmax) dmax = dd;
                    if (dd == 1) d1 = d1 + 1; else dbig = dbig + 1;
                    if (firstbad < 0) firstbad = i;
                    if (i % NFRAME == 0) f0bad = f0bad + 1; else fbad = fbad + 1;
                end

            if ($test$plusargs("dump") && vi == 0)
                $display("    fft_pow[0]=%012x fft_pow[1]=%012x  pw_mem[0][15:0]=%04x  dct0=%0d",
                         dbg_pw0, dbg_mel0[47:0], dbg_log0, $signed(dbg_dct0));
            $display("  %-10s %0d writes, %0d cycles; mismatches %0d (|d|=1 %0d, |d|>1 %0d, max %0d)",
                     names[vi], nwrite, cycles_o, errors, d1, dbig, dmax);
            $display("             first bad addr %0d (c%0d f%0d); frame0 bad %0d, frames1..100 bad %0d",
                     firstbad, (firstbad < 0) ? 0 : firstbad / NFRAME,
                     (firstbad < 0) ? 0 : firstbad % NFRAME, f0bad, fbad);
            if ($test$plusargs("dump") && vi == 0) begin
                $write("    frame0 rtl:");
                for (i = 0; i < 40; i = i + 1) $write(" %0d", got_mem[i*NFRAME]);
                $write("\n    frame0 ref:");
                for (i = 0; i < 40; i = i + 1) $write(" %0d", exp_mem[i*NFRAME]);
                $write("\n");
            end
            if (d1 + dbig > 0) errors_total = errors_total + d1 + dbig;
        end
    endtask

    initial begin
        names[0] = "0_silence"; names[1] = "1_yes"; names[2] = "2_stop";
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_mfcc.vcd");
            $dumpvars(0, tb_mfcc_top);
        end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        for (v = 0; v < NVEC; v = v + 1) run_vector(v);
        if (errors_total == 0)
            $display("MFCC_TOP: %0d clips x %0d codes bit-exact", NVEC, NCODE);
        else
            $display("MFCC_TOP: %0d code mismatches over %0d clips", errors_total, NVEC);
        $finish;
    end

endmodule
