// =============================================================================
// text_overlay.v -- draws NC 5x7 glyphs, scaled 4x, centred horizontally in the
//                   lower part of the panel.  Transparent background: pixels
//                   that are not part of a glyph pass the incoming colour on.
//
// Geometry is deliberately power-of-two so there is no divider anywhere:
//   SCALE = 4  -> each glyph pixel is 4x4, a glyph is 20x28
//   ADV   = 32 -> 5 bits: col = rel_x[9:5], gx = rel_x[4:2]
//   6 chars -> 192 px wide, centred at X0 = (480-192)/2 = 144, Y0 = 200
//
// Verilog string literals are LEFT aligned, so the leftmost character of a word
// sits in the most significant byte of `text`: col 0 -> text[47:40].
// =============================================================================
`timescale 1ns / 1ps

module text_overlay #(
    parameter integer NC     = 6,
    parameter integer X0     = 144,
    parameter integer Y0     = 200,
    parameter integer GW     = 5,
    parameter integer GH     = 7,
    parameter [23:0]  FG     = 24'hFFD23C,     // warm yellow
    parameter integer BOX_PAD = 4,              // plate margin around the glyphs
    parameter integer BOX_ON  = 1,              // 0 = transparent, text only
    // "translucent black": the plate subtracts (bg >> BOX_SHIFT) from each channel
    // instead of multiplying it down, so the graphic behind stays visible.
    //   1 -> 50% darker    (alpha 0.5)
    //   2 -> 25% darker    (alpha 0.25, default)
    //   3 -> 12.5% darker  (alpha 0.125, barely there)
    parameter integer BOX_SHIFT = 2
) (
    input  wire            clk,
    input  wire [9:0]      x,
    input  wire [8:0]      y,
    input  wire            show,
    input  wire [NC*8-1:0] text,
    input  wire [23:0]     rgb_in,
    output reg  [23:0]     rgb_out
);

    wire [9:0] rel_x = x - X0;                  // wraps for x < X0, which the
    wire [8:0] rel_y = y - Y0;                  // range tests below reject
    wire       in_x  = (x >= X0) && (x < X0 + NC*32);
    wire       in_y  = (y >= Y0) && (y < Y0 + GH*4);

    wire [4:0] col = rel_x[9:5];
    wire [2:0] gx  = rel_x[4:2];
    wire [2:0] gy  = rel_y[4:2];

    reg [7:0] ch;
    always @(*) begin
        case (col)
            5'd0: ch = text[47:40];
            5'd1: ch = text[39:32];
            5'd2: ch = text[31:24];
            5'd3: ch = text[23:16];
            5'd4: ch = text[15:8];
            default: ch = text[7:0];
        endcase
    end

    wire [7:0] bits;
    font_rom u_font (.ch(ch), .row(gy), .bits(bits));

    wire hit = show && in_x && in_y && (gx < GW) && (gy < GH) && bits[GW-1-gx];

    // ---- text plate ---------------------------------------------------------
    // A translucent strip behind the word: without it the yellow glyphs can land
    // on a bright part of the graphic and lose contrast.  This is a blend toward
    // black, not a solid plate -- a solid plate looks like an opaque black bar
    // over a dark background.
    wire in_box = show &&
                  (x >= X0 - BOX_PAD) && (x < X0 + NC*32 + BOX_PAD) &&
                  (y >= Y0 - BOX_PAD) && (y < Y0 + GH*4  + BOX_PAD);

    // per channel: c - (c >> BOX_SHIFT), i.e. c * (1 - 2^-BOX_SHIFT)
    wire [7:0] dim_r = rgb_in[23:16] - (rgb_in[23:16] >> BOX_SHIFT);
    wire [7:0] dim_g = rgb_in[15:8]  - (rgb_in[15:8]  >> BOX_SHIFT);
    wire [7:0] dim_b = rgb_in[7:0]   - (rgb_in[7:0]   >> BOX_SHIFT);
    wire [23:0] bg_dim = {dim_r, dim_g, dim_b};

    always @(posedge clk)
        rgb_out <= hit  ? FG
                 : (BOX_ON != 0 && in_box) ? bg_dim
                 : rgb_in;

endmodule
