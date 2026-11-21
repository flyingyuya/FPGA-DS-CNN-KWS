// =============================================================================
// accelerator_top.v -- layer FSM + one engine + one dual-bank memory.
//
//     map write (MFCC) --> FMEM (4096 x 8)          weight ROM, const ROM
//                              |                              |
//                              v                              v
//     +------------------------+------------------+   +--------+--------+
//     |            bank memory                    |   |  conv_engine    |
//     | 2 x KWS_BANK_WORDS words x 256 bit, 2 ports+-->+ (fetch FSM +    +--> argmax
//     | bank bit = address MSB (no read mux!)     |   |  address gen)   |    tree
//     +------------------------+------------------+   +--------+--------+      |
//                                                          |          class/margin
//                                                          v
//                                                  pe_array(32 lanes)
//
// Design notes
// ------------
// * ONE engine serves all 11 layers (see conv_engine.v).  The layer FSM here only
//   broadcasts a configuration and waits; it has no inner-loop logic.
//
// * Both ping-pong activation banks live in ONE memory whose top address bit is
//   the bank.  Layer N reads {src_sel, addr} and writes {dst_sel, addr}, so the
//   usual 256-bit read multiplexer between two banks disappears.  The reserved
//   zero word for padding comes for free: {src_sel, all-ones} is the last word of
//   the source bank.
//
// * Bank depth is KWS_BANK_WORDS = 2^ceil(log2(largest layer output)) -- 512 for
//   the 20x101 model (stem out 10x51 = 510), 1024 for 40x101.  That single
//   derived constant is what sets the activation BRAM count (8 here).
//
// * Two ways in: the stream loader (in_valid/in_data/in_last, used by the
//   accelerator testbench and by a host loading a vector) or map_we/map_addr/
//   map_data straight into FMEM (the on-chip MFCC).  They share FMEM's single
//   write port; `skip_load` says which path was used.
//
// * 27,035 cycles per inference (0.563 ms at 48 MHz) for the 20x101 model.
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module accelerator_top #(
    parameter integer LANES       = `KWS_LANES,
    parameter integer ACT_BITS    = `KWS_ACT_BITS,
    parameter integer W_BITS      = `KWS_W_BITS,
    parameter integer ACC_BITS    = `KWS_ACC_BITS,
    parameter integer SHIFT_BITS  = `KWS_SHIFT_BITS,
    parameter integer BANK_ADDR_W = `KWS_BANK_ADDR_W,
    parameter integer WROM_ADDR_W = `KWS_WROM_ADDR_W,
    parameter integer CH_ADDR_W   = `KWS_CH_ADDR_W,
    parameter integer FMEM_ADDR_W = `KWS_IN_ADDR_W,
    parameter integer DIM_W       = 8,
    parameter integer NUM_CLASSES = `KWS_NUM_CLASSES,
    parameter integer LOGIT_BITS  = 16,
    parameter         MEM_DIR     = "mem"
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- control -----------------------------------------------------------
    input  wire                          start,
    input  wire                          skip_load,   // frame already in FMEM (sim)
    output reg                           busy,
    output reg                           done,

    // ---- MFCC input stream -------------------------------------------------
    input  wire                          in_valid,
    input  wire [ACT_BITS-1:0]           in_data,
    input  wire                          in_last,
    output wire                          in_ready,

    // direct map write (on-chip MFCC), bypasses the stream loader
    input  wire                          map_we,
    input  wire [FMEM_ADDR_W-1:0]        map_addr,   // matches fmem's ADDR_W
    input  wire [ACT_BITS-1:0]           map_data,

    // ---- results -----------------------------------------------------------
    output wire [NUM_CLASSES*LOGIT_BITS-1:0] logits_o,
    output wire [3:0]                    class_o,
    output wire signed [LOGIT_BITS-1:0]  max_o,
    output wire signed [LOGIT_BITS-1:0]  margin_o,
    output reg                           result_valid,

    // ---- observability -----------------------------------------------------
    output reg  [31:0]                   cycles_o,
    output reg  [3:0]                    layer_o
);

    localparam [ACT_BITS-1:0] zp_in = `KWS_ZP_IN;
    localparam [15:0]         IN_PIXELS = `KWS_IN_PIXELS;

    // -------------------------------------------------------------------------
    // layer configuration (pure combinational decode)
    // -------------------------------------------------------------------------
    reg  [3:0] layer;
    wire [2:0] c_kind;
    wire [DIM_W-1:0] c_hi, c_wi, c_ho, c_wo, c_cin, c_cout, c_stride, c_pad,
                     c_ng, c_ng_in, c_ntaps;
    wire [9:0] c_wbase, c_ch_base;
    wire       c_last;

    layer_cfg #(.DIM_W(DIM_W)) u_cfg (
        .layer(layer), .kind(c_kind),
        .hi(c_hi), .wi(c_wi), .ho(c_ho), .wo(c_wo),
        .cin(c_cin), .cout(c_cout), .stride(c_stride), .pad(c_pad),
        .ng(c_ng), .ng_in(c_ng_in), .ntaps(c_ntaps),
        .wbase(c_wbase), .ch_base(c_ch_base), .last_layer(c_last));

    // bank assignment: the stem writes bank 0, then the banks alternate
    wire       dst_sel = layer[0];                       // even layer -> bank 0
    wire       src_sel = (layer == `KWS_L_STEM) ? 1'b0 : ~dst_sel;
    wire       stem_reads_fmem = (layer == `KWS_L_STEM);

    // -------------------------------------------------------------------------
    // memories
    // -------------------------------------------------------------------------
    wire        eng_act_re, eng_act_we;
    wire [BANK_ADDR_W-1:0] eng_act_raddr, eng_act_waddr;
    wire [LANES-1:0]       eng_act_wbe;
    wire [LANES*ACT_BITS-1:0] eng_act_wdata, bank_rdata;
    reg         load_start;
    wire        load_busy, load_done;
    wire        eng_fmem_re;
    wire [FMEM_ADDR_W-1:0] eng_fmem_addr;
    wire [ACT_BITS-1:0]    fmem_rdata;

    // one physical memory, bank bit is the address MSB
    act_bank #(.WORDS(2*`KWS_BANK_WORDS), .LANES(LANES), .ACT_BITS(ACT_BITS),
               .ADDR_W(BANK_ADDR_W+1)) u_bank (
        .clk    (clk),
        .we     (eng_act_we),
        .wbe    (eng_act_wbe),
        .waddr  ({dst_sel, eng_act_waddr}),
        .wdata  (eng_act_wdata),
        .re     (eng_act_re),
        .raddr  ({src_sel, eng_act_raddr}),
        .rdata  (bank_rdata));

    fmem #(.DEPTH(4096), .ADDR_W(FMEM_ADDR_W), .ACT_BITS(ACT_BITS),
           .INIT_FILE({MEM_DIR, "/input/mfcc_class00__silence_.mem"})) u_fmem (
        .clk(clk), .rst_n(rst_n),
        .ext_we(map_we), .ext_addr(map_addr), .ext_data(map_data),
        .load_start(load_start), .load_len(IN_PIXELS),
        .s_valid(in_valid), .s_data(in_data), .s_last(in_last),
        .s_ready(in_ready), .load_busy(load_busy), .load_done(load_done),
        .re(eng_fmem_re), .raddr(eng_fmem_addr), .rdata(fmem_rdata));

    wire eng_wrom_en;
    wire [WROM_ADDR_W-1:0] eng_wrom_addr;
    wire [LANES*W_BITS-1:0] wrom_data;

    weight_rom #(.WORDS(`KWS_WROM_WORDS), .LANES(LANES), .W_BITS(W_BITS),
                 .ADDR_W(WROM_ADDR_W), .INIT_FILE({MEM_DIR, "/weights/w_all.mem"})) u_wrom (
        .clk(clk), .en(eng_wrom_en), .addr(eng_wrom_addr), .rdata(wrom_data));

    wire ld_start, ld_done, ld_const, ld_busy_unused;
    wire [CH_ADDR_W-1:0] ld_ch_base;
    wire [4:0] ld_lane;
    wire signed [ACC_BITS-1:0] ld_bias;
    wire [SHIFT_BITS-1:0] ld_shift;

    const_loader #(.DEPTH(`KWS_CH_TOTAL), .ADDR_W(CH_ADDR_W),
                   .ACC_BITS(ACC_BITS), .SHIFT_BITS(SHIFT_BITS), .LANES(LANES),
                   .INIT_FILE({MEM_DIR, "/bias/b_all.mem"})) u_const (
        .clk(clk), .rst_n(rst_n), .start(ld_start), .ch_base(ld_ch_base),
        .busy(ld_busy_unused), .done(ld_done), .load_const(ld_const), .load_lane(ld_lane),
        .load_bias(ld_bias), .load_shift(ld_shift));

    // -------------------------------------------------------------------------
    // engine + PE array
    // -------------------------------------------------------------------------
    wire eng_busy, eng_done;
    reg  eng_start;
    wire pe_en, pe_init, pe_sel, pe_rd_sel;
    wire [LANES*W_BITS-1:0]  pe_w;
    wire [LANES*ACT_BITS-1:0] pe_a, pe_y;
    wire [LANES*LOGIT_BITS-1:0] pe_ys;
    wire [LANES*ACC_BITS-1:0] pe_acc_a, pe_acc_b;
    wire pe_ovf;
    wire [NUM_CLASSES*LOGIT_BITS-1:0] eng_logits;
    wire eng_logits_valid;

    conv_engine #(.LANES(LANES), .ACT_BITS(ACT_BITS), .W_BITS(W_BITS),
                  .ACC_BITS(ACC_BITS), .SHIFT_BITS(SHIFT_BITS),
                  .BANK_ADDR_W(BANK_ADDR_W), .FMEM_ADDR_W(FMEM_ADDR_W),
                  .WROM_ADDR_W(WROM_ADDR_W), .CH_ADDR_W(CH_ADDR_W),
                  .DIM_W(DIM_W), .LANE_AW(5),
                  .NUM_CLASSES(NUM_CLASSES), .LOGIT_BITS(LOGIT_BITS)) u_eng (
        .clk(clk), .rst_n(rst_n), .start(eng_start), .busy(eng_busy), .done(eng_done),
        .cfg_kind(c_kind),
        .cfg_hi(c_hi), .cfg_wi(c_wi), .cfg_ho(c_ho), .cfg_wo(c_wo),
        .cfg_cin(c_cin), .cfg_cout(c_cout), .cfg_stride(c_stride), .cfg_pad(c_pad),
        .cfg_ng(c_ng), .cfg_ng_in(c_ng_in), .cfg_ntaps(c_ntaps),
        .cfg_wbase(c_wbase), .cfg_ch_base(c_ch_base), .cfg_zp_in(zp_in),
        .ld_start(ld_start), .ld_ch_base(ld_ch_base), .ld_done(ld_done),
        .act_re(eng_act_re), .act_raddr(eng_act_raddr), .act_rdata(bank_rdata),
        .fmem_re(eng_fmem_re), .fmem_addr(eng_fmem_addr), .fmem_rdata(fmem_rdata),
        .wrom_en(eng_wrom_en), .wrom_addr(eng_wrom_addr), .wrom_data(wrom_data),
        .pe_en(pe_en), .pe_init(pe_init), .pe_sel(pe_sel), .pe_rd_sel(pe_rd_sel),
        .pe_w(pe_w), .pe_a(pe_a), .pe_y(pe_y), .pe_ys(pe_ys),
        .act_we(eng_act_we), .act_wbe(eng_act_wbe), .act_waddr(eng_act_waddr),
        .act_wdata(eng_act_wdata),
        .logits_o(eng_logits), .logits_valid(eng_logits_valid));

    pe_array #(.LANES(LANES), .W_BITS(W_BITS), .ACT_BITS(ACT_BITS),
               .ACC_BITS(ACC_BITS), .SHIFT_BITS(SHIFT_BITS)) u_pe (
        .clk(clk), .rst_n(rst_n), .load_const(ld_const), .load_lane(ld_lane),
        .load_bias(ld_bias), .load_shift(ld_shift),
        .en(pe_en), .init(pe_init), .sel(pe_sel), .rd_sel(pe_rd_sel),
        .w_word(pe_w), .a_word(pe_a), .y_word(pe_y), .ys_word(pe_ys),
        .acc_a_word(pe_acc_a), .acc_b_word(pe_acc_b), .ovf(pe_ovf));

    // -------------------------------------------------------------------------
    // classifier
    // -------------------------------------------------------------------------
    reg [NUM_CLASSES*LOGIT_BITS-1:0] logits_q;
    reg                              amax_start;
    assign logits_o = logits_q;

    wire amax_done;
    argmax_tree #(.NUM_CLASSES(NUM_CLASSES), .LOGIT_BITS(LOGIT_BITS)) u_argmax (
        .clk(clk), .rst_n(rst_n), .start(amax_start), .logits_i(logits_q),
        .class_o(class_o), .max_o(max_o), .margin_o(margin_o), .done(amax_done));

    // -------------------------------------------------------------------------
    // layer control
    // -------------------------------------------------------------------------
    localparam [2:0] S_IDLE  = 3'd0,
                     S_LOAD  = 3'd1,
                     S_CFG   = 3'd2,
                     S_RUN   = 3'd3,
                     S_NEXT  = 3'd4,
                     S_AMAX  = 3'd5,
                     S_DONE  = 3'd6;

    reg [2:0]  state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            layer        <= 4'd0;
            busy         <= 1'b0;
            done         <= 1'b0;
            result_valid <= 1'b0;
            load_start   <= 1'b0;
            eng_start    <= 1'b0;
            logits_q     <= {NUM_CLASSES*LOGIT_BITS{1'b0}};
            cycles_o     <= 32'd0;
            layer_o      <= 4'd0;
        end else begin
            eng_start  <= 1'b0;
            load_start <= 1'b0;
            amax_start <= (state == S_NEXT) && (layer == `KWS_NUM_LAYERS - 1);
            done       <= 1'b0;

            if (busy) cycles_o <= cycles_o + 32'd1;
            if (eng_logits_valid) logits_q <= eng_logits;

            case (state)
            S_IDLE: begin
                if (start) begin
                    busy         <= 1'b1;
                    result_valid <= 1'b0;
                    layer        <= 4'd0;
                    layer_o      <= 4'd0;
                    cycles_o     <= 32'd0;
                    if (skip_load) begin
                        state <= S_CFG;
                    end else begin
                        load_start <= 1'b1;
                        state      <= S_LOAD;
                    end
                end
            end

            S_LOAD: begin
                if (load_done) state <= S_CFG;
            end

            S_CFG: begin
                layer_o   <= layer;
                eng_start <= 1'b1;
                state     <= S_RUN;
            end

            S_RUN: begin
                if (eng_done) state <= S_NEXT;
            end

            S_NEXT: begin
                if (layer == `KWS_NUM_LAYERS - 1) begin
                    state <= S_AMAX;          // 12 cycles, one comparison each
                end else begin
                    layer <= layer + 4'd1;
                    state <= S_CFG;
                end
            end

            S_AMAX: begin
                if (amax_done) state <= S_DONE;   // amax_start is a 1-cycle pulse
            end

            S_DONE: begin
                busy         <= 1'b0;
                done         <= 1'b1;
                result_valid <= 1'b1;
                state        <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
