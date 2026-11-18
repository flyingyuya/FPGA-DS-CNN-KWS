// =============================================================================
// tb_fft_core.v -- bit-exact testbench for the 512-point real FFT + power stage.
//
// The RTL is the Sonic-Speed-Parallelism fft_core adapted to N=512 (its address
// generator and butterfly are already parameterised) with the output stage
// changed from "|X| ~= max + 0.4375*min" to "|X|^2 = re^2 + im^2", because the
// MFCC model was trained on log power and an approximate magnitude would bias
// every frequency bin upwards by up to 0.75 dB.
//
// Golden data: scripts/golden/gen_fft_pow_vectors.py drives scripts/golden/
// fft_dsp.py, a bit-level model of the same butterfly (twiddle rounding, per-stage
// halving, saturation) that reads the same twiddle ROM, so the comparison below
// is exact -- not "within a tolerance".
//
// Vectors: impulse, single tone, multi-tone, noise, and the two extreme frames
// that stress saturation.
//
// Run from the project root:  bash sim/run.sh fft
// =============================================================================
`timescale 1ns / 1ps

module tb_fft_core;

    localparam integer N     = 512;
    localparam integer LOG2N = 9;
    localparam integer DW    = 24;
    localparam integer NOUT  = N / 2;             // 256 bins: DC .. bin 255
    localparam integer NVEC  = 6;
    localparam real    CLK_NS = 20.833;           // 48 MHz

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    reg  signed [DW-1:0] x_in = 0;
    reg                  x_valid = 1'b0;
    wire [LOG2N-2:0]     y_index;
    wire [2*DW:0]        y_pow;
    wire                 y_valid;
    wire                 busy;
    wire [7:0]           drop_cnt;

    fft_core #(.N(N), .LOG2N(LOG2N), .DW(DW), .TW(16)) u_fft (
        .clk(clk), .rst_n(rst_n),
        .x_in(x_in), .x_valid(x_valid),
        .y_index(y_index), .y_pow(y_pow), .y_valid(y_valid),
        .busy(busy), .drop_cnt(drop_cnt));

    // ---- golden data -------------------------------------------------------
    reg [8*24-1:0]        names [0:NVEC-1];
    reg [DW-1:0]          x_mem [0:N-1];
    reg [2*DW:0]          exp_mem [0:NOUT-1];
    reg [2*DW:0]          got [0:NOUT-1];

    integer errors = 0;
    integer v, i, got_n;
    integer cycles = 0;
    reg [8*64-1:0] path;

    initial begin
        names[0] = "impulse";  names[1] = "sine8";    names[2] = "multitone";
        names[3] = "noise";    names[4] = "max_pos";  names[5] = "max_neg";
    end

    // ---- collect outputs ---------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && y_valid) begin
            got[y_index] <= y_pow;
            got_n = got_n + 1;
        end
        if (busy) cycles = cycles + 1;
    end

    task run_vector;
        input integer vi;
        begin
            $sformat(path, "golden/fft/in_%0s.mem", names[vi]);
            $readmemh(path, x_mem);
            $sformat(path, "golden/fft/exp_%0s.mem", names[vi]);
            $readmemh(path, exp_mem);

            got_n  = 0;
            cycles = 0;
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                x_valid = 1'b1;
                x_in    = x_mem[i];
            end
            @(negedge clk);
            x_valid = 1'b0;
            // wait for all 256 bins (bounded: the tie to `busy` alone can stall)
            for (i = 0; i < 40000 && got_n < NOUT; i = i + 1) @(negedge clk);
            $display("    got %0d/%0d bins after %0d cycles", got_n, NOUT, cycles);
            @(negedge clk);

            for (i = 0; i < NOUT; i = i + 1) begin
                if (got[i] !== exp_mem[i]) begin
                    errors = errors + 1;
                    if (errors < 8)
                        $display("    FAIL %0s bin %0d: got %013x expected %013x",
                                 names[vi], i, got[i], exp_mem[i]);
                end
            end
            $display("  %-9s %0d bins, %0d cycles", names[vi], got_n, cycles);
        end
    endtask

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_fft.vcd");
            $dumpvars(0, tb_fft_core);
        end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        for (v = 0; v < NVEC; v = v + 1) run_vector(v);

        if (drop_cnt !== 8'd0) begin
            errors = errors + 1;
            $display("  FAIL: input FIFO dropped %0d samples", drop_cnt);
        end
        if (errors == 0)
            $display("FFT_CORE: %0d vectors x %0d bins bit-exact", NVEC, NOUT);
        else
            $display("FFT_CORE: %0d mismatches", errors);
        $finish;
    end

endmodule
