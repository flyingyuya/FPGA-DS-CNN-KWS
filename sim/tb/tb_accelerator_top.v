// =============================================================================
// tb_accelerator_top.v -- end-to-end testbench: MFCC frame in, classification out.
//
// Runs all 14 golden vectors (one per class plus the two extreme frames) through
// the complete accelerator and compares the class index and all 12 logit codes
// with golden/top_class.mem and golden/top_logits.mem.
//
// Two load paths are exercised:
//   * vector 0 goes through the MFCC stream port (valid/ready/last), which is how
//     the host or an on-chip front end will fill FMEM;
//   * the remaining vectors preload FMEM and use `skip_load`, which keeps the
//     simulation short without weakening the layer-sequencing coverage.
//
// Run from the project root:  bash sim/run.sh top [+vcd]
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module tb_accelerator_top;

    localparam integer NUM_CLASSES = `KWS_NUM_CLASSES;
    localparam integer NVEC        = 14;
    localparam integer IN_PIXELS   = `KWS_IN_PIXELS;
    localparam real    CLK_NS      = 20.833;      // 48 MHz

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    reg         start = 1'b0;
    reg         skip_load = 1'b0;
    wire        busy, done;
    reg         in_valid = 1'b0;
    reg  [7:0]  in_data = 8'd0;
    reg         in_last = 1'b0;
    wire        in_ready;
    wire [NUM_CLASSES*16-1:0] logits_o;
    wire [3:0]  class_o;
    wire signed [15:0] max_o, margin_o;
    wire        result_valid;
    wire [31:0] cycles_o;
    wire [3:0]  layer_o;

    accelerator_top #(.NUM_CLASSES(NUM_CLASSES), .MEM_DIR("mem")) u_accel (
        .clk(clk), .rst_n(rst_n),
        .start(start), .skip_load(skip_load), .busy(busy), .done(done),
        .in_valid(in_valid), .in_data(in_data), .in_last(in_last), .in_ready(in_ready),
        .logits_o(logits_o), .class_o(class_o), .max_o(max_o), .margin_o(margin_o),
        .result_valid(result_valid), .cycles_o(cycles_o), .layer_o(layer_o));

    // ---- golden data -------------------------------------------------------
    reg [8*40-1:0] vec_name [0:NVEC-1];
    reg [7:0]      exp_class [0:NVEC-1];
    reg [23:0]     exp_logit [0:NVEC*NUM_CLASSES-1];
    reg [7:0]      frame [0:4095];

    integer errors = 0;
    integer v, i, c;
    integer total_cycles = 0;
    reg [8*64-1:0] path;

    task run_vector;
        input integer vi;
        input         use_stream;
        begin
            if (use_stream) begin
                // push the frame through the port, like a host would
                @(negedge clk);
                start = 1'b1; skip_load = 1'b0;
                @(negedge clk);
                start = 1'b0;
                // honour the handshake: the loader only becomes ready a couple of
                // cycles after `start`, and anything driven before that is lost
                while (!in_ready) @(negedge clk);
                for (i = 0; i < IN_PIXELS; i = i + 1) begin
                    in_valid = 1'b1;
                    in_data  = frame[i];
                    in_last  = (i == IN_PIXELS - 1);
                    @(negedge clk);
                end
                in_valid = 1'b0;
                in_last  = 1'b0;
            end else begin
                // preload FMEM directly (equivalent to having received the frame)
                $readmemh(path, u_accel.u_fmem.mem);
                @(negedge clk);
                start = 1'b1; skip_load = 1'b1;
                @(negedge clk);
                start = 1'b0; skip_load = 1'b0;
            end
            while (!done) @(negedge clk);
            @(negedge clk);
        end
    endtask

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_accel.vcd");
            $dumpvars(0, tb_accelerator_top);
        end

        // The vector order matches golden/top_vectors.mem (kept for reference;
        // $readmemh cannot read strings, so the names live here).
        vec_name[0] = "class00__silence_";
        vec_name[1] = "class01__unknown_";
        vec_name[2] = "class02_yes";
        vec_name[3] = "class03_no";
        vec_name[4] = "class04_up";
        vec_name[5] = "class05_down";
        vec_name[6] = "class06_left";
        vec_name[7] = "class07_right";
        vec_name[8] = "class08_on";
        vec_name[9] = "class09_off";
        vec_name[10] = "class10_stop";
        vec_name[11] = "class11_go";
        vec_name[12] = "edge_all_min";
        vec_name[13] = "edge_all_max";
        $readmemh("golden/top_class.mem", exp_class);
        $readmemh("golden/top_logits.mem", exp_logit);

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        for (v = 0; v < NVEC; v = v + 1) begin
            $sformat(path, "mem/input/mfcc_%0s.mem", vec_name[v]);
            $readmemh(path, frame);

            run_vector(v, (v == 0));

            // ---- compare -----------------------------------------------------
            if (class_o !== exp_class[v][3:0]) begin
                errors = errors + 1;
                $display("  FAIL %0s: class %0d expected %0d",
                         vec_name[v], class_o, exp_class[v]);
            end
            for (c = 0; c < NUM_CLASSES; c = c + 1) begin
                if ($signed(logits_o[c*16 +: 16]) !== $signed(exp_logit[v*NUM_CLASSES + c][15:0])) begin
                    errors = errors + 1;
                    if (errors < 12)
                        $display("  FAIL %0s logit %0d: got %0d expected %0d",
                                 vec_name[v], c, $signed(logits_o[c*16 +: 16]),
                                 $signed(exp_logit[v*NUM_CLASSES + c][15:0]));
                end
            end
            total_cycles = total_cycles + cycles_o;
            $display("  %-18s -> class %2d  max %5d  margin %5d  cycles %0d",
                     vec_name[v], class_o, $signed(max_o), $signed(margin_o), cycles_o);
        end

        $display("  average %0d cycles  (= %0.3f ms at 48 MHz)",
                 total_cycles / NVEC, (total_cycles / NVEC) / 48.0e3);
        if (errors == 0)
            $display("TOP: all %0d vectors classified correctly (class + 12 logits bit-exact)", NVEC);
        else
            $display("TOP: %0d mismatches", errors);
        $finish;
    end

endmodule
