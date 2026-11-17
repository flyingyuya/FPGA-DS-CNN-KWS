// =============================================================================
// layer_cfg.v -- the layer FSM's configuration table (Fig. 2 of the paper).
//
// A pure combinational decode of the layer index into every field the engine
// needs, so the engine itself never knows which layer it is running.  Adding a
// layer to the network means regenerating ../rtl/kws_params.vh and adding one
// branch here -- the datapath does not change.
//
// The order of the branches IS the network, and it must match LAYERS in
// sim/reference_model.py, which is asserted by export_mem.py.
// =============================================================================
`timescale 1ns / 1ps
`include "kws_params.vh"

module layer_cfg #(
    parameter integer DIM_W = 8
) (
    input  wire [3:0]        layer,
    output reg  [2:0]        kind,
    output reg  [DIM_W-1:0]  hi, wi, ho, wo, cin, cout, stride, pad, ng, ng_in, ntaps,
    output reg  [9:0]        wbase,
    output reg  [9:0]        ch_base,
    output reg               last_layer
);

    localparam [2:0] K_STD = 3'd0, K_DW = 3'd1, K_PW = 3'd2, K_GAP = 3'd3, K_FC = 3'd4;

    always @* begin
        // defaults: a safe, inert configuration
        kind = K_PW; hi = 1; wi = 1; ho = 1; wo = 1;
        cin = 1; cout = 1; stride = 1; pad = 0; ng = 1; ng_in = 1; ntaps = 1;
        wbase = 0; ch_base = 0;
        last_layer = (layer == `KWS_NUM_LAYERS - 1);

        case (layer)
        `KWS_L_STEM: begin
            kind = K_STD;
            hi = `KWS_STEM_HI; wi = `KWS_STEM_WI; ho = `KWS_STEM_HO; wo = `KWS_STEM_WO;
            cin = `KWS_STEM_CIN; cout = `KWS_STEM_COUT;
            stride = `KWS_STEM_STRIDE; pad = `KWS_STEM_PAD;
            ng = `KWS_STEM_NG; ng_in = `KWS_STEM_CIN/`KWS_LANES; ntaps = 9;
            wbase = `KWS_STEM_WROM_BASE; ch_base = `KWS_STEM_CH_BASE;
        end
        `KWS_L_DW1: begin
            kind = K_DW;
            hi = `KWS_DW1_HI; wi = `KWS_DW1_WI; ho = `KWS_DW1_HO; wo = `KWS_DW1_WO;
            cin = `KWS_DW1_CIN; cout = `KWS_DW1_COUT;
            stride = `KWS_DW1_STRIDE; pad = `KWS_DW1_PAD;
            ng = `KWS_DW1_NG; ng_in = `KWS_DW1_CIN/`KWS_LANES; ntaps = 9;
            wbase = `KWS_DW1_WROM_BASE; ch_base = `KWS_DW1_CH_BASE;
        end
        `KWS_L_PW1: begin
            kind = K_PW;
            hi = `KWS_PW1_HI; wi = `KWS_PW1_WI; ho = `KWS_PW1_HO; wo = `KWS_PW1_WO;
            cin = `KWS_PW1_CIN; cout = `KWS_PW1_COUT;
            stride = 1; pad = 0;
            ng = `KWS_PW1_NG; ng_in = `KWS_PW1_CIN/`KWS_LANES; ntaps = 1;
            wbase = `KWS_PW1_WROM_BASE; ch_base = `KWS_PW1_CH_BASE;
        end
        `KWS_L_DW2: begin
            kind = K_DW;
            hi = `KWS_DW2_HI; wi = `KWS_DW2_WI; ho = `KWS_DW2_HO; wo = `KWS_DW2_WO;
            cin = `KWS_DW2_CIN; cout = `KWS_DW2_COUT;
            stride = `KWS_DW2_STRIDE; pad = `KWS_DW2_PAD;
            ng = `KWS_DW2_NG; ng_in = `KWS_DW2_CIN/`KWS_LANES; ntaps = 9;
            wbase = `KWS_DW2_WROM_BASE; ch_base = `KWS_DW2_CH_BASE;
        end
        `KWS_L_PW2: begin
            kind = K_PW;
            hi = `KWS_PW2_HI; wi = `KWS_PW2_WI; ho = `KWS_PW2_HO; wo = `KWS_PW2_WO;
            cin = `KWS_PW2_CIN; cout = `KWS_PW2_COUT;
            stride = 1; pad = 0;
            ng = `KWS_PW2_NG; ng_in = `KWS_PW2_CIN/`KWS_LANES; ntaps = 1;
            wbase = `KWS_PW2_WROM_BASE; ch_base = `KWS_PW2_CH_BASE;
        end
        `KWS_L_DW3: begin
            kind = K_DW;
            hi = `KWS_DW3_HI; wi = `KWS_DW3_WI; ho = `KWS_DW3_HO; wo = `KWS_DW3_WO;
            cin = `KWS_DW3_CIN; cout = `KWS_DW3_COUT;
            stride = `KWS_DW3_STRIDE; pad = `KWS_DW3_PAD;
            ng = `KWS_DW3_NG; ng_in = `KWS_DW3_CIN/`KWS_LANES; ntaps = 9;
            wbase = `KWS_DW3_WROM_BASE; ch_base = `KWS_DW3_CH_BASE;
        end
        `KWS_L_PW3: begin
            kind = K_PW;
            hi = `KWS_PW3_HI; wi = `KWS_PW3_WI; ho = `KWS_PW3_HO; wo = `KWS_PW3_WO;
            cin = `KWS_PW3_CIN; cout = `KWS_PW3_COUT;
            stride = 1; pad = 0;
            ng = `KWS_PW3_NG; ng_in = `KWS_PW3_CIN/`KWS_LANES; ntaps = 1;
            wbase = `KWS_PW3_WROM_BASE; ch_base = `KWS_PW3_CH_BASE;
        end
        `KWS_L_DW4: begin
            kind = K_DW;
            hi = `KWS_DW4_HI; wi = `KWS_DW4_WI; ho = `KWS_DW4_HO; wo = `KWS_DW4_WO;
            cin = `KWS_DW4_CIN; cout = `KWS_DW4_COUT;
            stride = `KWS_DW4_STRIDE; pad = `KWS_DW4_PAD;
            ng = `KWS_DW4_NG; ng_in = `KWS_DW4_CIN/`KWS_LANES; ntaps = 9;
            wbase = `KWS_DW4_WROM_BASE; ch_base = `KWS_DW4_CH_BASE;
        end
        `KWS_L_PW4: begin
            kind = K_PW;
            hi = `KWS_PW4_HI; wi = `KWS_PW4_WI; ho = `KWS_PW4_HO; wo = `KWS_PW4_WO;
            cin = `KWS_PW4_CIN; cout = `KWS_PW4_COUT;
            stride = 1; pad = 0;
            ng = `KWS_PW4_NG; ng_in = `KWS_PW4_CIN/`KWS_LANES; ntaps = 1;
            wbase = `KWS_PW4_WROM_BASE; ch_base = `KWS_PW4_CH_BASE;
        end
        `KWS_L_GAP: begin
            kind = K_GAP;
            hi = `KWS_GAP_HI; wi = `KWS_GAP_WI; ho = 1; wo = 1;
            cin = `KWS_GAP_CIN; cout = `KWS_GAP_COUT;
            stride = 1; pad = 0;
            ng = `KWS_GAP_NG; ng_in = `KWS_GAP_CIN/`KWS_LANES; ntaps = `KWS_GAP_N;
            wbase = 0; ch_base = `KWS_GAP_CH_BASE;
        end
        `KWS_L_FC: begin
            kind = K_FC;
            hi = 1; wi = 1; ho = 1; wo = 1;
            cin = `KWS_FC_CIN; cout = `KWS_FC_COUT;
            stride = 1; pad = 0;
            ng = `KWS_FC_NG; ng_in = `KWS_FC_CIN/`KWS_LANES; ntaps = 1;
            wbase = `KWS_FC_WROM_BASE; ch_base = `KWS_FC_CH_BASE;
        end
        default: ;
        endcase
    end

endmodule
