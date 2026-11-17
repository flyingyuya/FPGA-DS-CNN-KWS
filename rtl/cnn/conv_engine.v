// =============================================================================
// conv_engine.v -- ONE configurable engine that serves every layer of the DS-CNN.
//
// This is the accelerator's fetch FSM + address generator.
// The layer FSM lives in the top and only broadcasts a configuration;
// every layer type is the same loop nest with different knobs:
//
//   for g   in 0..NG-1                     # output channel group = 32 lanes
//     for oy, ox                           # output pixels
//       for ic in 0..N_IC-1                # input channels (1 for dw / pool)
//         for tap in 0..N_TAPS-1           # 9 (3x3), 1 (1x1), or the pool length
//
//   kind   N_TAPS  N_IC  activation          weight addr         output stage
//   -------------------------------------------------------------------------
//   STD    9       cin   FMEM, byte bcast    base + ic*9 + tap   shift + clamp
//   DW     9       1     bank, whole word    base + g*9 + tap    shift + clamp
//   PW     1       cin   bank, byte bcast    base + g*cin + ic   shift + clamp
//   GAP    N       1     bank, whole word    constant 1          shift + clamp
//   FC     1       cin   bank, byte bcast    base + ic           shift, no clamp
//
// Three reasons one engine is enough:
//   * the pointwise layer uses lane = OUTPUT channel with a broadcast activation,
//     the same lane semantics as the standard convolution;
//   * the global average pool is a depthwise pass with unit weights, N taps,
//     bias 2**(G-1) and shift G -- no module and no divider;
//   * the fully connected layer is a pointwise layer with a single pixel.
//
// ---------------------------------------------------------------------------
// Addresses are incremental, there is no runtime multiplier
// ---------------------------------------------------------------------------
//   row_ptr        += stride*wi    per output row     (stride is 1 or 2 -> shift)
//   ox_s           += stride       per output column
//   row_pix_act/out+= wo*ng_*      per output row
//   ox_act/out     += ng_*         per output column
//   base_g         += N_IC*N_TAPS  per channel group  (1, 9, cin -> shift+add)
//   t_off           = ky*wi         from a 3:1 mux, no multiplication
//   w_off          += 1            per MAC, reset per job
//
// Padding needs no 256-bit data mask: an out-of-range tap redirects the bank
// address to a reserved all-zero word.  The stem is the one layer whose zero
// point is non-zero, and it reads FMEM instead, where the byte path substitutes
// zp_in for out-of-range taps.
//
// Pipeline:
//   cycle t   : issue addresses (memory ports combinational from the counters)
//   cycle t+1 : data valid, PE multiplies and accumulates
//   cycle t+2 : if that was a job's last tap, drain the bank and write it back
// =============================================================================
`timescale 1ns / 1ps

module conv_engine #(
    parameter integer LANES       = 32,
    parameter integer ACT_BITS    = 8,
    parameter integer W_BITS      = 8,
    parameter integer ACC_BITS    = 24,
    parameter integer SHIFT_BITS  = 4,
    parameter integer BANK_ADDR_W = 10,
    parameter integer FMEM_ADDR_W = 12,
    parameter integer WROM_ADDR_W = 10,
    parameter integer CH_ADDR_W   = 10,
    parameter integer DIM_W       = 8,
    parameter integer LANE_AW     = 5,
    parameter integer NUM_CLASSES = 12,
    parameter integer LOGIT_BITS  = 16
) (
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire                          start,
    output reg                           busy,
    output reg                           done,

    // ---- layer configuration ----------------------------------------------
    input  wire [2:0]                    cfg_kind,
    input  wire [DIM_W-1:0]              cfg_hi,
    input  wire [DIM_W-1:0]              cfg_wi,
    input  wire [DIM_W-1:0]              cfg_ho,
    input  wire [DIM_W-1:0]              cfg_wo,
    input  wire [DIM_W-1:0]              cfg_cin,
    input  wire [DIM_W-1:0]              cfg_cout,
    input  wire [DIM_W-1:0]              cfg_stride,
    input  wire [DIM_W-1:0]              cfg_pad,
    input  wire [DIM_W-1:0]              cfg_ng,        // ceil(cout/LANES)
    input  wire [DIM_W-1:0]              cfg_ng_in,     // ceil(cin/LANES)
    input  wire [DIM_W-1:0]              cfg_ntaps,     // pool length (GAP only)
    input  wire [WROM_ADDR_W-1:0]        cfg_wbase,
    input  wire [CH_ADDR_W-1:0]          cfg_ch_base,
    input  wire [ACT_BITS-1:0]           cfg_zp_in,     // non-zero for the stem

    // ---- constant loader ---------------------------------------------------
    output reg                           ld_start,
    output reg  [CH_ADDR_W-1:0]          ld_ch_base,
    input  wire                          ld_done,

    // ---- activation bank (source) -----------------------------------------
    output wire                          act_re,
    output wire [BANK_ADDR_W-1:0]        act_raddr,
    input  wire [LANES*ACT_BITS-1:0]     act_rdata,

    // ---- MFCC input memory (std only) -------------------------------------
    output wire                          fmem_re,
    output wire [FMEM_ADDR_W-1:0]        fmem_addr,
    input  wire [ACT_BITS-1:0]           fmem_rdata,

    // ---- weight ROM --------------------------------------------------------
    output wire                          wrom_en,
    output wire [WROM_ADDR_W-1:0]        wrom_addr,
    input  wire [LANES*W_BITS-1:0]       wrom_data,

    // ---- PE array ----------------------------------------------------------
    output wire                          pe_en,
    output wire                          pe_init,
    output wire                          pe_sel,
    output wire                          pe_rd_sel,
    output wire [LANES*W_BITS-1:0]       pe_w,
    output wire [LANES*ACT_BITS-1:0]     pe_a,
    input  wire [LANES*ACT_BITS-1:0]     pe_y,
    input  wire [LANES*LOGIT_BITS-1:0]   pe_ys,

    // ---- activation bank (destination) ------------------------------------
    output wire                          act_we,
    output wire [LANES-1:0]              act_wbe,
    output wire [BANK_ADDR_W-1:0]        act_waddr,
    output wire [LANES*ACT_BITS-1:0]     act_wdata,

    // ---- classifier output (FC only) --------------------------------------
    output reg  [NUM_CLASSES*LOGIT_BITS-1:0] logits_o,
    output reg                           logits_valid
);

    localparam integer PIX_W = 12;                     // 40*101 = 4040 positions
    localparam integer ADD_W = 14;                     // signed address accumulator
    localparam [BANK_ADDR_W-1:0] PAD_WORD = {BANK_ADDR_W{1'b1}};   // reserved zero word
    // unit weight, replicated across all lanes: the pool multiplies by 1
    localparam [W_BITS-1:0] W_ONE = {{(W_BITS-1){1'b0}}, 1'b1};

    localparam [2:0] K_STD = 3'd0, K_DW = 3'd1, K_PW = 3'd2, K_GAP = 3'd3, K_FC = 3'd4;

    localparam [2:0] S_IDLE  = 3'd0,
                     S_LOAD  = 3'd1,
                     S_RUN   = 3'd2,
                     S_FLUSH = 3'd3,
                     S_DONE  = 3'd4;

    reg [2:0] state;

    // ---- kind decode -------------------------------------------------------
    wire is_gap = (cfg_kind == K_GAP);
    wire is_std = (cfg_kind == K_STD);
    wire is_fc  = (cfg_kind == K_FC);
    wire word_mode = (cfg_kind == K_DW) || is_gap;
    wire ic_is_one = (cfg_kind == K_DW) || is_gap;

    // ---- counters ----------------------------------------------------------
    reg [DIM_W-1:0] g, oy, ox, ic, ky, kx;
    reg [DIM_W-1:0] ng_r, ng_in_r, n_taps_r, n_ic_r, cout_r;
    reg [DIM_W-1:0] hi_r, wi_r, ho_r, wo_r, stride_r, pad_r;
    reg [PIX_W-1:0] lin_tap;
    reg [ADD_W-1:0] wi_x2;

    reg [ADD_W-1:0] row_ptr;          // (oy*stride - pad) * wi
    reg [ADD_W-1:0] step_row;         // stride * wi
    reg [DIM_W-1:0] oy_s, ox_s;       // oy*stride, ox*stride
    reg [ADD_W-1:0] pix_addr0;        // row_ptr + ox_s - pad  (job base, flat)
    reg [PIX_W-1:0] row_pix_act, ox_act, row_pix_out, ox_out;
    reg [PIX_W-1:0] row_step_act, row_step_out;
    reg [BANK_ADDR_W-1:0] gap_addr;

    reg [WROM_ADDR_W-1:0] base_g, w_off, group_step;
    reg job_par;

    // ---- pipeline stage B (MAC) / C (drain) --------------------------------
    reg          b_valid, b_init, b_sel, b_last, b_inrange;
    reg [LANE_AW-1:0] b_byte;
    reg [BANK_ADDR_W-1:0] b_out_addr;
    reg [5:0]    b_nvalid;
    reg          d_valid, d_sel;
    reg [BANK_ADDR_W-1:0] d_addr;
    reg [5:0]    d_nvalid;

    // -------------------------------------------------------------------------
    // Address generation
    // -------------------------------------------------------------------------
    // tap offset inside the 3x3 window: ky*wi from a 3:1 mux (no multiplier)
    wire [ADD_W-1:0] t_off = (ky == 2'd0) ? {ADD_W{1'b0}}
                           : (ky == 2'd1) ? $signed({1'b0, wi_r})
                           : wi_x2;
    wire signed [ADD_W-1:0] pixel_addr = pix_addr0 + t_off + $signed({2'b0, kx});

    wire signed [DIM_W:0] iy = $signed({1'b0, oy_s}) + $signed({1'b0, ky}) - $signed({1'b0, pad_r});
    wire signed [DIM_W:0] ix = $signed({1'b0, ox_s}) + $signed({1'b0, kx}) - $signed({1'b0, pad_r});
    wire in_range = (iy >= 0) && (iy < $signed({1'b0, hi_r})) &&
                    (ix >= 0) && (ix < $signed({1'b0, wi_r}));

    wire run_conv = (state == S_RUN) && !is_gap;
    wire run_gap  = (state == S_RUN) && is_gap;

    // depthwise: flat pixel address scaled by NG_in, plus the channel group
    wire [PIX_W-1:0] dw_addr = (ng_in_r == 8'd2) ? {pixel_addr[PIX_W-2:0], 1'b0}
                                                 : pixel_addr[PIX_W-1:0];
    // pointwise / fully connected: the output pixel, group selected by ic/32
    wire [PIX_W-1:0] pw_addr = row_pix_act + ox_act + ic[DIM_W-1:LANE_AW];

    assign act_re    = run_gap || (run_conv && !is_std);
    assign act_raddr = is_gap  ? (gap_addr + g)
                     : word_mode ? ((in_range ? dw_addr[PIX_W-1:0] + g : PAD_WORD))
                     : pw_addr[BANK_ADDR_W-1:0];

    assign fmem_re   = run_conv && is_std;
    assign fmem_addr = pixel_addr[FMEM_ADDR_W-1:0];

    assign wrom_en   = run_conv;                      // the pool uses unit weights
    assign wrom_addr = base_g + w_off;

    // -------------------------------------------------------------------------
    // PE data path
    // -------------------------------------------------------------------------
    wire [ACT_BITS-1:0] bank_byte = act_rdata[b_byte*ACT_BITS +: ACT_BITS];
    // FMEM's read is registered, so the range flag that belongs to the data on
    // the bus is the one from the *previous* cycle.  Using the combinational flag
    // here mixes the current tap's range with the previous tap's data.
    wire [ACT_BITS-1:0] fmem_byte = b_inrange ? fmem_rdata : cfg_zp_in;
    wire [ACT_BITS-1:0] a_bcast   = is_std ? fmem_byte : bank_byte;

    assign pe_en     = b_valid;
    assign pe_init   = b_init;
    assign pe_sel    = b_sel;
    assign pe_rd_sel = d_sel;
    assign pe_w      = is_gap ? {LANES{W_ONE}} : wrom_data;
    assign pe_a      = word_mode ? act_rdata : {LANES{a_bcast}};

    // -------------------------------------------------------------------------
    // Write back
    // -------------------------------------------------------------------------
    assign act_we    = d_valid && !is_fc;
    assign act_waddr = d_addr;
    assign act_wdata = pe_y;
    assign act_wbe   = (d_nvalid >= LANES) ? {LANES{1'b1}}
                                           : ((32'd1 << d_nvalid) - 32'd1);

    wire [DIM_W-1:0] g32       = g << LANE_AW;
    wire [DIM_W-1:0] cout_left = cout_r - g32;
    wire [5:0]       nvalid    = (cout_left > LANES) ? LANES[5:0] : cout_left[5:0];

    // "last tap of the job" as a wire so both places agree
    wire tap_last = (n_taps_r == 8'd1) ? 1'b1 : ((kx == 8'd2) && (ky == 8'd2));
    wire job_last = tap_last && (ic == n_ic_r - 1'b1);

    integer c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            g <= 0; oy <= 0; ox <= 0; ic <= 0; ky <= 0; kx <= 0; lin_tap <= 0;
            ng_r <= 0; ng_in_r <= 0; n_taps_r <= 0; n_ic_r <= 0; cout_r <= 0;
            hi_r <= 0; wi_r <= 0; ho_r <= 0; wo_r <= 0; stride_r <= 0; pad_r <= 0;
            wi_x2 <= 0; row_ptr <= 0; step_row <= 0; oy_s <= 0; ox_s <= 0;
            pix_addr0 <= 0; row_pix_act <= 0; ox_act <= 0;
            row_pix_out <= 0; ox_out <= 0; row_step_act <= 0; row_step_out <= 0;
            gap_addr <= 0; base_g <= 0; w_off <= 0; group_step <= 0; job_par <= 1'b0;
            ld_start <= 1'b0; ld_ch_base <= 0;
            b_valid <= 0; b_init <= 0; b_sel <= 0; b_last <= 0; b_inrange <= 0; b_byte <= 0;
            b_out_addr <= 0; b_nvalid <= 0;
            d_valid <= 0; d_sel <= 0; d_addr <= 0; d_nvalid <= 0;
            logits_o <= 0; logits_valid <= 1'b0;
        end else begin
            done         <= 1'b0;
            ld_start     <= 1'b0;
            logits_valid <= 1'b0;
            b_valid      <= 1'b0;

            // ---- stage C ---------------------------------------------------
            d_valid <= 1'b0;
            if (b_valid && b_last) begin
                d_valid  <= 1'b1;
                d_sel    <= b_sel;
                d_addr   <= b_out_addr;
                d_nvalid <= b_nvalid;
            end
            if (d_valid && is_fc) begin
                for (c = 0; c < NUM_CLASSES; c = c + 1)
                    logits_o[c*LOGIT_BITS +: LOGIT_BITS] <= pe_ys[c*LOGIT_BITS +: LOGIT_BITS];
                logits_valid <= 1'b1;
            end

            case (state)
            // -----------------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    busy <= 1'b1;
                    g <= 0; oy <= 0; ox <= 0; ic <= 0; ky <= 0; kx <= 0; lin_tap <= 0;
                    ng_r     <= cfg_ng;
                    ng_in_r  <= cfg_ng_in;
                    cout_r   <= cfg_cout;
                    hi_r     <= cfg_hi;
                    wi_r     <= cfg_wi;
                    ho_r     <= cfg_ho;
                    wo_r     <= cfg_wo;
                    stride_r <= cfg_stride;
                    pad_r    <= cfg_pad;
                    n_ic_r   <= ic_is_one ? 8'd1 : cfg_cin;
                    n_taps_r <= is_gap ? cfg_ntaps
                              : ((cfg_kind == K_STD || cfg_kind == K_DW) ? 8'd9 : 8'd1);
                    wi_x2     <= {1'b0, cfg_wi} + {1'b0, cfg_wi};
                    step_row  <= (cfg_stride == 8'd2) ? $signed({1'b0, cfg_wi, 1'b0})
                                                      : $signed({1'b0, cfg_wi});
                    row_ptr   <= (cfg_pad != 0) ? -$signed({1'b0, cfg_wi}) : {ADD_W{1'b0}};
                    // flat address of the job's top-left tap: -pad*wi - pad
                    pix_addr0 <= (cfg_pad != 0) ? -($signed({1'b0, cfg_wi}) + 1'b1)
                                                : {ADD_W{1'b0}};
                    oy_s <= 0; ox_s <= 0;
                    row_pix_act <= 0; ox_act <= 0; row_pix_out <= 0; ox_out <= 0;
                    row_step_act <= (cfg_ng_in == 8'd2) ? {1'b0, cfg_wo, 1'b0} : cfg_wo;
                    row_step_out <= (cfg_ng    == 8'd2) ? {1'b0, cfg_wo, 1'b0} : cfg_wo;
                    base_g     <= cfg_wbase;
                    w_off      <= 0;
                    group_step <= (cfg_kind == K_DW)  ? 10'd9
                                : (cfg_kind == K_STD) ? ((cfg_cin << 3) + cfg_cin)
                                : cfg_cin;
                    gap_addr  <= 0;
                    job_par   <= 1'b0;
                    ld_start  <= 1'b1;
                    ld_ch_base <= cfg_ch_base;
                    state     <= S_LOAD;
                end
            end

            // -----------------------------------------------------------------
            S_LOAD: begin
                if (ld_done) state <= S_RUN;
            end

            // -----------------------------------------------------------------
            S_RUN: begin
                b_valid    <= 1'b1;
                b_init     <= (ic == 0) && (ky == 0) && (kx == 0) && (lin_tap == 0);
                b_sel      <= job_par;
                b_last     <= is_gap ? (lin_tap == n_taps_r - 1'b1) : job_last;
                b_inrange  <= is_gap ? 1'b1 : in_range;
                b_byte     <= ic[LANE_AW-1:0];
                b_out_addr <= is_gap ? g : (row_pix_out + ox_out + g);
                b_nvalid   <= nvalid;
                w_off      <= w_off + 1'b1;

                if (is_gap) begin
                    // one linear pass over the source tensor, unit weights
                    if (lin_tap == n_taps_r - 1'b1) begin
                        lin_tap  <= 0;
                        gap_addr <= 0;
                        job_par  <= ~job_par;
                        state    <= S_FLUSH;
                    end else begin
                        lin_tap  <= lin_tap + 1'b1;
                        gap_addr <= gap_addr + ng_in_r;
                    end
                end else begin
                    // ---- tap nest: ic (outer) x (ky,kx) (inner) --------------
                    if (tap_last) begin
                        kx <= 0;
                        ky <= 0;
                        if (ic == n_ic_r - 1'b1) begin
                            // job finished: next output pixel
                            ic      <= 0;
                            w_off   <= 0;
                            job_par <= ~job_par;
                            if (ox == wo_r - 1'b1) begin
                                ox      <= 0;
                                ox_s    <= 0;
                                ox_act  <= 0;
                                ox_out  <= 0;
                                if (oy == ho_r - 1'b1) begin
                                    oy    <= 0;
                                    state <= S_FLUSH;
                                end else begin
                                    oy          <= oy + 1'b1;
                                    oy_s        <= oy_s + stride_r;
                                    row_ptr     <= row_ptr + step_row;
                                    pix_addr0   <= row_ptr + step_row - $signed({1'b0, pad_r});
                                    row_pix_act <= row_pix_act + row_step_act;
                                    row_pix_out <= row_pix_out + row_step_out;
                                end
                            end else begin
                                ox        <= ox + 1'b1;
                                ox_s      <= ox_s + stride_r;
                                pix_addr0 <= pix_addr0 + $signed({1'b0, stride_r});
                                ox_act    <= ox_act + ng_in_r;
                                ox_out    <= ox_out + ng_r;
                            end
                        end else begin
                            ic <= ic + 1'b1;
                        end
                    end else if (kx == 8'd2) begin
                        kx <= 0;
                        ky <= ky + 1'b1;
                    end else begin
                        kx <= kx + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------------
            S_FLUSH: begin
                // both pipeline stages must be empty before the constants change:
                // the drain of the last job of the group still needs the old
                // bias/shift registers
                if (!b_valid && !d_valid) begin
                    if (g == ng_r - 1'b1) begin
                        state <= S_DONE;
                    end else begin
                        g          <= g + 1'b1;
                        base_g     <= base_g + group_step;
                        w_off      <= 0;
                        oy         <= 0;
                        ox         <= 0;
                        ic         <= 0;
                        ky         <= 0;
                        kx         <= 0;
                        lin_tap    <= 0;
                        gap_addr   <= 0;
                        oy_s       <= 0;
                        ox_s       <= 0;
                        row_ptr    <= (pad_r != 0) ? -$signed({1'b0, wi_r}) : {ADD_W{1'b0}};
                        pix_addr0  <= (pad_r != 0) ? -($signed({1'b0, wi_r}) + 1'b1)
                                                   : {ADD_W{1'b0}};
                        row_pix_act <= 0;
                        ox_act      <= 0;
                        row_pix_out <= 0;
                        ox_out      <= 0;
                        ld_start   <= 1'b1;
                        ld_ch_base <= ld_ch_base + LANES;
                        state      <= S_LOAD;
                    end
                end
            end

            S_DONE: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
