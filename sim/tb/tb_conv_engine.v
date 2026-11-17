// =============================================================================
// tb_conv_engine.v -- layer testbench for the unified convolution engine.
//
// Runs every layer type through the same engine and compares the result with the
// golden images from sim/reference_model.py:
//
//   std  stem   FMEM -> bank0     9 taps, stride 2, padding      1020 words
//   dw   dw1..dw4                 whole-word activation            260/130 words
//   pw   pw1..pw4                 broadcast activation             520/130 words
//   gap               bank -> 2 codes, unit weights, bias 32, shift 6
//   fc                12 logit codes via the unclamped PE output
//
// The point of the test is that ONE engine produces all of these: if the config
// decode, the tap nest, the incremental address generators or the padding
// redirect were wrong for any layer type, that layer would mismatch.
//
// Run from the project root:  bash sim/run.sh conv [+trace] [+vcd]
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module tb_conv_engine;

    localparam integer LANES       = `KWS_LANES;
    localparam integer ACT_BITS    = `KWS_ACT_BITS;
    localparam integer W_BITS      = `KWS_W_BITS;
    localparam integer ACC_BITS    = `KWS_ACC_BITS;
    localparam integer SHIFT_BITS  = `KWS_SHIFT_BITS;
    localparam integer BANK_ADDR_W = `KWS_BANK_ADDR_W;
    localparam integer FMEM_ADDR_W = 12;
    localparam integer WROM_ADDR_W = `KWS_WROM_ADDR_W;
    localparam integer CH_ADDR_W   = `KWS_CH_ADDR_W;
    localparam integer NUM_CLASSES = `KWS_NUM_CLASSES;
    localparam integer NWORDS      = `KWS_BANK_WORDS;
    localparam real    CLK_NS      = 20.833;      // 48 MHz

    localparam [2:0] K_STD = 3'd0, K_DW = 3'd1, K_PW = 3'd2, K_GAP = 3'd3, K_FC = 3'd4;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    // ---- configuration ----------------------------------------------------
    reg                start = 1'b0;
    wire               busy, done;
    reg  [2:0]         cfg_kind = K_PW;
    reg  [7:0]         cfg_hi = 0, cfg_wi = 0, cfg_ho = 0, cfg_wo = 0;
    reg  [7:0]         cfg_cin = 0, cfg_cout = 0, cfg_stride = 1, cfg_pad = 0;
    reg  [7:0]         cfg_ng = 1, cfg_ng_in = 1, cfg_ntaps = 1;
    reg  [WROM_ADDR_W-1:0] cfg_wbase = 0;
    reg  [CH_ADDR_W-1:0]   cfg_ch_base = 0;

    // ---- engine <-> memories ----------------------------------------------
    wire               ld_start, ld_done;
    wire [CH_ADDR_W-1:0] ld_ch_base;
    wire               ld_const;
    wire [4:0]         ld_lane;
    wire signed [ACC_BITS-1:0] ld_bias;
    wire [SHIFT_BITS-1:0]      ld_shift;

    wire               act_re, act_we, fmem_re, wrom_en;
    wire [BANK_ADDR_W-1:0] act_raddr, act_waddr;
    wire [LANES*ACT_BITS-1:0] act_rdata, act_wdata;
    wire [LANES-1:0]   act_wbe;
    wire [FMEM_ADDR_W-1:0] fmem_addr;
    wire [ACT_BITS-1:0]    fmem_rdata;
    wire [WROM_ADDR_W-1:0] wrom_addr;
    wire [LANES*W_BITS-1:0] wrom_data;

    wire               pe_en, pe_init, pe_sel, pe_rd_sel;
    wire [LANES*W_BITS-1:0]  pe_w;
    wire [LANES*ACT_BITS-1:0] pe_a, pe_y;
    wire [LANES*16-1:0]      pe_ys;
    wire [ACT_BITS-1:0]      zp_in = `KWS_ZP_IN;
    wire [LANES*ACC_BITS-1:0] pe_acc_a, pe_acc_b;
    wire               pe_ovf;

    wire [NUM_CLASSES*16-1:0] logits_o;
    wire               logits_valid;

    // ---- memories ----------------------------------------------------------
    act_bank #(.WORDS(NWORDS), .LANES(LANES), .ACT_BITS(ACT_BITS),
               .ADDR_W(BANK_ADDR_W)) u_src (
        .clk(clk), .we(1'b0), .wbe({LANES{1'b0}}), .waddr({BANK_ADDR_W{1'b0}}),
        .wdata({LANES*ACT_BITS{1'b0}}), .re(act_re), .raddr(act_raddr),
        .rdata(act_rdata));

    act_bank #(.WORDS(NWORDS), .LANES(LANES), .ACT_BITS(ACT_BITS),
               .ADDR_W(BANK_ADDR_W)) u_dst (
        .clk(clk), .we(act_we), .wbe(act_wbe), .waddr(act_waddr), .wdata(act_wdata),
        .re(1'b0), .raddr({BANK_ADDR_W{1'b0}}), .rdata());

    // MFCC input memory (8-bit, registered read like a BRAM)
    reg [ACT_BITS-1:0] fmem_mem [0:4095];
    integer fk;
    initial for (fk = 0; fk < 4096; fk = fk + 1) fmem_mem[fk] = {ACT_BITS{1'b0}};
    reg [ACT_BITS-1:0] fmem_d;
    always @(posedge clk) if (fmem_re) fmem_d <= fmem_mem[fmem_addr];
    assign fmem_rdata = fmem_d;

    weight_rom #(.WORDS(`KWS_WROM_WORDS), .LANES(LANES), .W_BITS(W_BITS),
                 .ADDR_W(WROM_ADDR_W), .INIT_FILE("mem/weights/w_all.mem")) u_wrom (
        .clk(clk), .en(wrom_en), .addr(wrom_addr), .rdata(wrom_data));

    const_loader #(.DEPTH(`KWS_CH_TOTAL), .ADDR_W(CH_ADDR_W),
                   .ACC_BITS(ACC_BITS), .SHIFT_BITS(SHIFT_BITS), .LANES(LANES),
                   .INIT_FILE("mem/bias/b_all.mem")) u_const (
        .clk(clk), .rst_n(rst_n), .start(ld_start), .ch_base(ld_ch_base),
        .busy(), .done(ld_done), .load_const(ld_const), .load_lane(ld_lane),
        .load_bias(ld_bias), .load_shift(ld_shift));

    pe_array #(.LANES(LANES), .W_BITS(W_BITS), .ACT_BITS(ACT_BITS),
               .ACC_BITS(ACC_BITS), .SHIFT_BITS(SHIFT_BITS)) u_pe (
        .clk(clk), .rst_n(rst_n), .load_const(ld_const), .load_lane(ld_lane),
        .load_bias(ld_bias), .load_shift(ld_shift),
        .en(pe_en), .init(pe_init), .sel(pe_sel), .rd_sel(pe_rd_sel),
        .w_word(pe_w), .a_word(pe_a), .y_word(pe_y), .ys_word(pe_ys),
        .acc_a_word(pe_acc_a), .acc_b_word(pe_acc_b), .ovf(pe_ovf));

    conv_engine #(.LANES(LANES), .ACT_BITS(ACT_BITS), .W_BITS(W_BITS),
                  .ACC_BITS(ACC_BITS), .SHIFT_BITS(SHIFT_BITS),
                  .BANK_ADDR_W(BANK_ADDR_W), .FMEM_ADDR_W(FMEM_ADDR_W),
                  .WROM_ADDR_W(WROM_ADDR_W), .CH_ADDR_W(CH_ADDR_W),
                  .DIM_W(8), .LANE_AW(5), .NUM_CLASSES(NUM_CLASSES),
                  .LOGIT_BITS(16)) u_eng (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
        .cfg_kind(cfg_kind),
        .cfg_hi(cfg_hi), .cfg_wi(cfg_wi), .cfg_ho(cfg_ho), .cfg_wo(cfg_wo),
        .cfg_cin(cfg_cin), .cfg_cout(cfg_cout), .cfg_stride(cfg_stride),
        .cfg_pad(cfg_pad), .cfg_ng(cfg_ng), .cfg_ng_in(cfg_ng_in),
        .cfg_ntaps(cfg_ntaps), .cfg_wbase(cfg_wbase), .cfg_ch_base(cfg_ch_base),
        .cfg_zp_in(zp_in),
        .ld_start(ld_start), .ld_ch_base(ld_ch_base), .ld_done(ld_done),
        .act_re(act_re), .act_raddr(act_raddr), .act_rdata(act_rdata),
        .fmem_re(fmem_re), .fmem_addr(fmem_addr), .fmem_rdata(fmem_rdata),
        .wrom_en(wrom_en), .wrom_addr(wrom_addr), .wrom_data(wrom_data),
        .pe_en(pe_en), .pe_init(pe_init), .pe_sel(pe_sel), .pe_rd_sel(pe_rd_sel),
        .pe_w(pe_w), .pe_a(pe_a), .pe_y(pe_y), .pe_ys(pe_ys),
        .act_we(act_we), .act_wbe(act_wbe), .act_waddr(act_waddr),
        .act_wdata(act_wdata),
        .logits_o(logits_o), .logits_valid(logits_valid));

    // ---- bookkeeping -------------------------------------------------------
    reg [LANES*ACT_BITS-1:0] exp_mem [0:NWORDS-1];
    integer errors = 0, cycles = 0, layers_run = 0;
    integer i, w, lane, c;
    reg [23:0] exp_code [0:63];

    integer err0;
    always @(posedge clk)
        if ($test$plusargs("gaptrace") && u_eng.is_gap && u_eng.state == 3'd2 &&
            u_eng.lin_tap < 5)
            $display("   [%0d] gap tap=%0d addr=%0d en=%b init=%b a0=%02x w0=%02x accA=%06x g=%0d",
                     cycles, u_eng.lin_tap, act_raddr, pe_en, pe_init,
                     pe_a[7:0], pe_w[7:0], pe_acc_a[23:0], u_eng.g);
    always @(posedge clk)
        if ($test$plusargs("taptrace") && cycles < 46 && rst_n)
            $display("   [%0d] st=%0d en=%b init=%b a0=%02x w0=%02x inr=%b fmem=%0d fad=%0d pad=%0d",
                     cycles, u_eng.state, pe_en, pe_init, pe_a[7:0], pe_w[7:0],
                     u_eng.in_range, fmem_re, fmem_addr, u_eng.pad_r);
    // ---- first-write trace (+wtrace) --------------------------------------
    integer nw = 0;
    always @(posedge clk) begin
        if ($test$plusargs("wtrace") && act_we) begin
            nw = nw + 1;
            if (nw < 6)
                $display("      [%0d] write addr=%0d y0=%02x a0=%02x w0=%02x accA=%06x sh0=%0d bias0=%06x inr=%b",
                         cycles, act_waddr, pe_y[7:0], pe_a[7:0], pe_w[7:0],
                         pe_acc_a[23:0], u_pe.shift_q[0], u_pe.bias_q[0],
                         u_eng.in_range);
        end
    end

    task check_bank;
        input integer nwords;
        input integer budget;
        begin
            err0 = errors;
            for (w = 0; w < nwords; w = w + 1)
                if (u_dst.mem[w] !== exp_mem[w]) begin
                    errors = errors + 1;
                    if (errors < 6) begin
                        $display("    FAIL word %0d", w);
                        for (lane = 0; lane < LANES; lane = lane + 1)
                            if (u_dst.mem[w][lane*ACT_BITS +: ACT_BITS] !==
                                exp_mem[w][lane*ACT_BITS +: ACT_BITS])
                                $display("      lane %2d got %02x expected %02x", lane,
                                         u_dst.mem[w][lane*ACT_BITS +: ACT_BITS],
                                         exp_mem[w][lane*ACT_BITS +: ACT_BITS]);
                    end
                end
            $display("  %0d words checked, %0d cycles (budget %0d, %0d%%), %0d mismatched",
                     nwords, cycles, budget, (100 * (cycles - budget)) / budget, errors - err0);
            if (errors > err0 && $test$plusargs("dump")) begin
                for (lane = 0; lane < 8; lane = lane + 1)
                    $display("    w0 lane %2d got %02x expected %02x",
                             lane, u_dst.mem[0][lane*ACT_BITS +: ACT_BITS],
                             exp_mem[0][lane*ACT_BITS +: ACT_BITS]);
            end
        end
    endtask

    task run;                                   // pulse start, wait for done
        begin
            cycles = 0;
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            while (!done && cycles < 400000) begin
                @(negedge clk);
                cycles = cycles + 1;
            end
            layers_run = layers_run + 1;
        end
    endtask

    task clear_dst;
        begin
            for (i = 0; i < NWORDS; i = i + 1) u_dst.mem[i] = {LANES*ACT_BITS{1'b0}};
        end
    endtask

    // bank -> bank layer (std / dw / pw)
    task run_bank_layer;
        input [8*64-1:0] src_file, exp_file;
        input [2:0]      kind;
        input integer    hi, wi, ho, wo, cin, cout, stride, pad, ng, ng_in,
                         wbase, ch_base, nwords, budget;
        begin
            clear_dst;
            if (kind == K_STD) $readmemh(src_file, fmem_mem);
            else               $readmemh(src_file, u_src.mem);
            $readmemh(exp_file, exp_mem);
            cfg_kind = kind; cfg_hi = hi; cfg_wi = wi; cfg_ho = ho; cfg_wo = wo;
            cfg_cin = cin; cfg_cout = cout; cfg_stride = stride; cfg_pad = pad;
            cfg_ng = ng; cfg_ng_in = ng_in; cfg_ntaps = 9;
            cfg_wbase = wbase; cfg_ch_base = ch_base;
            run;
            check_bank(nwords, budget);
        end
    endtask

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_conv_engine.vcd");
            $dumpvars(0, tb_conv_engine);
        end
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        // ---- standard convolution: the stem reads FMEM ---------------------
        $display("=== stem (std, 9 taps, stride 2, %0d words) ===", `KWS_STEM_OUT_WORDS);
        run_bank_layer("mem/input/mfcc_class00__silence_.mem",
                       "golden/act_stem_class00__silence_.mem", K_STD,
                       `KWS_STEM_HI, `KWS_STEM_WI, `KWS_STEM_HO, `KWS_STEM_WO,
                       `KWS_STEM_CIN, `KWS_STEM_COUT, `KWS_STEM_STRIDE, `KWS_STEM_PAD,
                       `KWS_STEM_NG, 1, `KWS_STEM_WROM_BASE, `KWS_STEM_CH_BASE,
                       `KWS_STEM_OUT_WORDS, 9180);

        // ---- depthwise ------------------------------------------------------
        $display("=== dw1 (32ch, stride 2) ===");
        run_bank_layer("golden/act_stem_class00__silence_.mem",
                       "golden/act_dw1_class00__silence_.mem", K_DW,
                       `KWS_DW1_HI, `KWS_DW1_WI, `KWS_DW1_HO, `KWS_DW1_WO,
                       `KWS_DW1_CIN, `KWS_DW1_COUT, `KWS_DW1_STRIDE, `KWS_DW1_PAD,
                       `KWS_DW1_NG, `KWS_DW1_CIN/`KWS_LANES, `KWS_DW1_WROM_BASE,
                       `KWS_DW1_CH_BASE, `KWS_DW1_OUT_WORDS, 2340);

        $display("=== dw2 (32ch, stride 2) ===");
        run_bank_layer("golden/act_pw1_class00__silence_.mem",
                       "golden/act_dw2_class00__silence_.mem", K_DW,
                       `KWS_DW2_HI, `KWS_DW2_WI, `KWS_DW2_HO, `KWS_DW2_WO,
                       `KWS_DW2_CIN, `KWS_DW2_COUT, `KWS_DW2_STRIDE, `KWS_DW2_PAD,
                       `KWS_DW2_NG, 1, `KWS_DW2_WROM_BASE, `KWS_DW2_CH_BASE,
                       `KWS_DW2_OUT_WORDS, 585);

        $display("=== dw3 (64ch, stride 1) ===");
        run_bank_layer("golden/act_pw2_class00__silence_.mem",
                       "golden/act_dw3_class00__silence_.mem", K_DW,
                       `KWS_DW3_HI, `KWS_DW3_WI, `KWS_DW3_HO, `KWS_DW3_WO,
                       `KWS_DW3_CIN, `KWS_DW3_COUT, `KWS_DW3_STRIDE, `KWS_DW3_PAD,
                       `KWS_DW3_NG, 2, `KWS_DW3_WROM_BASE, `KWS_DW3_CH_BASE,
                       `KWS_DW3_OUT_WORDS, 1170);

        $display("=== dw4 (64ch, stride 1) ===");
        run_bank_layer("golden/act_pw3_class00__silence_.mem",
                       "golden/act_dw4_class00__silence_.mem", K_DW,
                       `KWS_DW4_HI, `KWS_DW4_WI, `KWS_DW4_HO, `KWS_DW4_WO,
                       `KWS_DW4_CIN, `KWS_DW4_COUT, `KWS_DW4_STRIDE, `KWS_DW4_PAD,
                       `KWS_DW4_NG, 2, `KWS_DW4_WROM_BASE, `KWS_DW4_CH_BASE,
                       `KWS_DW4_OUT_WORDS, 1170);

        // ---- pointwise ------------------------------------------------------
        $display("=== pw1 (32->32, 10x26) ===");
        run_bank_layer("golden/act_dw1_class00__silence_.mem",
                       "golden/act_pw1_class00__silence_.mem", K_PW,
                       `KWS_PW1_HI, `KWS_PW1_WI, `KWS_PW1_HO, `KWS_PW1_WO,
                       `KWS_PW1_CIN, `KWS_PW1_COUT, 1, 0,
                       `KWS_PW1_NG, 1, `KWS_PW1_WROM_BASE, `KWS_PW1_CH_BASE,
                       `KWS_PW1_OUT_WORDS, 8320);

        $display("=== pw2 (32->64, 5x13) ===");
        run_bank_layer("golden/act_dw2_class00__silence_.mem",
                       "golden/act_pw2_class00__silence_.mem", K_PW,
                       `KWS_PW2_HI, `KWS_PW2_WI, `KWS_PW2_HO, `KWS_PW2_WO,
                       `KWS_PW2_CIN, `KWS_PW2_COUT, 1, 0,
                       `KWS_PW2_NG, 1, `KWS_PW2_WROM_BASE, `KWS_PW2_CH_BASE,
                       `KWS_PW2_OUT_WORDS, 4160);

        $display("=== pw3 (64->64, 5x13) ===");
        run_bank_layer("golden/act_dw3_class00__silence_.mem",
                       "golden/act_pw3_class00__silence_.mem", K_PW,
                       `KWS_PW3_HI, `KWS_PW3_WI, `KWS_PW3_HO, `KWS_PW3_WO,
                       `KWS_PW3_CIN, `KWS_PW3_COUT, 1, 0,
                       `KWS_PW3_NG, 2, `KWS_PW3_WROM_BASE, `KWS_PW3_CH_BASE,
                       `KWS_PW3_OUT_WORDS, 8320);

        $display("=== pw4 (64->64, 5x13) ===");
        run_bank_layer("golden/act_dw4_class00__silence_.mem",
                       "golden/act_pw4_class00__silence_.mem", K_PW,
                       `KWS_PW4_HI, `KWS_PW4_WI, `KWS_PW4_HO, `KWS_PW4_WO,
                       `KWS_PW4_CIN, `KWS_PW4_COUT, 1, 0,
                       `KWS_PW4_NG, 2, `KWS_PW4_WROM_BASE, `KWS_PW4_CH_BASE,
                       `KWS_PW4_OUT_WORDS, 8320);

        // ---- global average pool: 64 codes from 2 words --------------------
        $display("=== gap (65 taps, unit weights) ===");
        clear_dst;
        $readmemh("golden/act_pw4_class00__silence_.mem", u_src.mem);
        for (i = 0; i < 64; i = i + 1) exp_code[i] = 24'h0;
        $readmemh("golden/act_gap_class00__silence_.mem", exp_code);
        cfg_kind = K_GAP; cfg_hi = `KWS_GAP_HI; cfg_wi = `KWS_GAP_WI;
        cfg_ho = 1; cfg_wo = 1; cfg_cin = `KWS_GAP_CIN; cfg_cout = `KWS_GAP_COUT;
        cfg_stride = 1; cfg_pad = 0; cfg_ng = 2; cfg_ng_in = 2;
        cfg_ntaps = `KWS_GAP_N; cfg_wbase = 0; cfg_ch_base = `KWS_GAP_CH_BASE;
        run;
        for (i = 0; i < 64; i = i + 1) begin
            // lane i%32 of word i/32
            if (u_dst.mem[i/32][(i%32)*ACT_BITS +: ACT_BITS] !== exp_code[i][ACT_BITS-1:0]) begin
                errors = errors + 1;
                if (errors < 6)
                    $display("    FAIL gap code %0d: got %02x expected %02x", i,
                             u_dst.mem[i/32][(i%32)*ACT_BITS +: ACT_BITS],
                             exp_code[i][ACT_BITS-1:0]);
            end
        end
        $display("  64 codes checked, %0d cycles (budget 130)", cycles);

        // ---- fully connected: 12 logits through the unclamped output --------
        $display("=== fc (64 -> 12) ===");
        for (i = 0; i < 2; i = i + 1) u_src.mem[i] = u_dst.mem[i];   // pool result
        for (i = 0; i < NUM_CLASSES; i = i + 1) exp_code[i] = 24'h0;
        $readmemh("golden/act_fc_class00__silence_.mem", exp_code);
        cfg_kind = K_FC; cfg_hi = 1; cfg_wi = 1; cfg_ho = 1; cfg_wo = 1;
        cfg_cin = `KWS_FC_CIN; cfg_cout = `KWS_FC_COUT; cfg_stride = 1; cfg_pad = 0;
        cfg_ng = 1; cfg_ng_in = 2; cfg_ntaps = 1;
        cfg_wbase = `KWS_FC_WROM_BASE; cfg_ch_base = `KWS_FC_CH_BASE;
        run;
        for (i = 0; i < NUM_CLASSES; i = i + 1) begin
            if ($signed(logits_o[i*16 +: 16]) !== $signed(exp_code[i][15:0])) begin
                errors = errors + 1;
                if (errors < 6)
                    $display("    FAIL logit %0d: got %0d expected %0d", i,
                             $signed(logits_o[i*16 +: 16]), $signed(exp_code[i][15:0]));
            end
        end
        $display("  12 logits checked, %0d cycles (budget 64)", cycles);

        if (pe_ovf) begin
            errors = errors + 1;
            $display("FAIL: accumulator overflow flag raised");
        end
        if (errors == 0)
            $display("CONV ENGINE: all %0d layers match the golden model", layers_run);
        else
            $display("CONV ENGINE: %0d mismatches", errors);
        $finish;
    end

endmodule
