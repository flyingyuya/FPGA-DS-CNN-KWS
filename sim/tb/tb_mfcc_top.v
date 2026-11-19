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
    wire        dbg_we;
    wire [1:0]  dbg_stage;
    wire [11:0] dbg_addr;
    wire [47:0] dbg_data;
    wire [7:0]  frame_o;

    mfcc_top #(.NSAMP(NSAMP), .NFRAME(NFRAME), .SKIP_LOAD(1)) u_mfcc (
        .clk(clk), .rst_n(rst_n),
        .s_valid(1'b0), .s_data(16'sd0), .s_ready(),
        .start(start), .busy(busy), .done(done),
        .mfcc_we(mfcc_we), .mfcc_addr(mfcc_addr), .mfcc_data(mfcc_data),
        .cycles_o(cycles_o), .frame_o(frame_o),
        .dbg_we(dbg_we), .dbg_stage(dbg_stage), .dbg_addr(dbg_addr), .dbg_data(dbg_data));

    reg [8*24-1:0] names [0:NVEC-1];
    reg [7:0]      exp_mem [0:NCODE-1];
    reg [7:0]      got_mem [0:NCODE-1];
    reg [8*64-1:0] path;

    integer errors = 0, i, v, nwrite, dd, d1, dbig, dmax;
    integer errors_total = 0, firstbad, f0bad, fbad, nfbig;

    // (a hierarchical probe of the mel engine's power buffer reads X under
    //  iverilog even though the engine consumes it -- the useful diagnostic is the
    //  |d| histogram below, plus comparing intermediate stage dumps from
    //  scripts/golden/mfcc_ref.py)

    // race-free trace: register everything, then print one cycle later
    reg        ff_we_d;
    reg [9:0]  t_idx_d;
    reg [13:0] ridx_d;
    reg signed [15:0] samp_q_d, wprod_d;
    reg [7:0]  frame_d;
    always @(posedge clk) begin
        ff_we_d  <= u_mfcc.fft_we;
        t_idx_d  <= u_mfcc.t_idx;
        ridx_d   <= u_mfcc.ridx;
        samp_q_d <= u_mfcc.samp_q;
        wprod_d  <= u_mfcc.wprod;
        frame_d  <= u_mfcc.frame;
    end
    always @(posedge clk) begin
        if (rst_n && v == 0 && ff_we_d && frame_d == 8'd1 && t_idx_d < 10'd4)
            $display("    trc f=%0d t=%0d ridx=%0d sampq=%0d wprod=%0d",
                     frame_d, t_idx_d, ridx_d, samp_q_d, wprod_d);
    end

    integer fd;
    always @(posedge clk) if (rst_n && dbg_we)
        $fwrite(fd, "%0d %0d %0h\n", dbg_stage, dbg_addr, dbg_data);

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

            errors = 0; d1 = 0; dbig = 0; dmax = 0; firstbad = -1; f0bad = 0; fbad = 0; nfbig = 0;
            for (i = 0; i < NCODE; i = i + 1)
                if (got_mem[i] !== exp_mem[i]) begin
                    errors = errors + 1;
                    dd = got_mem[i] - exp_mem[i];
                    if (dd < 0) dd = -dd;
                    if (dd > dmax) dmax = dd;
                    if (dd == 1) d1 = d1 + 1; else dbig = dbig + 1;
                    if (firstbad < 0) firstbad = i;
                    if (i % NFRAME == 0) f0bad = f0bad + 1;
                    else begin fbad = fbad + 1; if (dd > 1) nfbig = nfbig + 1; end
                end

            $display("  %-10s %0d writes, %0d cycles; mismatches %0d (|d|=1 %0d, |d|>1 %0d, max %0d)",
                     names[vi], nwrite, cycles_o, errors, d1, dbig, dmax);
            $display("             frames 1..100: %0d bad, of which |d|>1: %0d", fbad, nfbig);
            $display("             first bad addr %0d (c%0d f%0d); frame0 bad %0d, frames1..100 bad %0d",
                     firstbad, (firstbad < 0) ? 0 : firstbad / NFRAME,
                     (firstbad < 0) ? 0 : firstbad % NFRAME, f0bad, fbad);
            // counters are plain regs (safe to read hierarchically; only BRAM
            // arrays return X under iverilog)
            $display("             fft_drop=%0d bins=%0d frame=%0d",
                     u_mfcc.fft_drop, u_mfcc.bins, u_mfcc.frame_o);
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
        fd = $fopen("sim/build/mfcc_stage.txt", "w");
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        for (v = 0; v < NVEC; v = v + 1) run_vector(v);
        $fclose(fd);
        if (errors_total == 0)
            $display("MFCC_TOP: %0d clips x %0d codes bit-exact", NVEC, NCODE);
        else
            $display("MFCC_TOP: %0d code mismatches over %0d clips", errors_total, NVEC);
        $finish;
    end

endmodule
