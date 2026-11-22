// =============================================================================
// word_to_text.v -- recognised class -> 6 ASCII characters, held on screen for
//                   HOLD_CYC pixel-clock cycles (1 s).
//
// WHY THIS RUNS IN THE PIXEL DOMAIN
//   The first version lived in the 48 MHz domain and pushed a 1-second LEVEL plus
//   a 48-bit payload through a 2-FF synchroniser.  It simulated fine but never
//   lit a pixel on hardware.  Only ONE bit crosses now (a toggle), and the whole
//   hold timer runs locally:
//     - `tog` flips on every classification (48 MHz domain, stable for a second)
//     - a 3-FF sync catches the flip, an edge detector turns it into a 1-cycle
//       event, the class nibble is sampled on that event, the counter reloads.
//   A toggle is immune to pulse-width questions: the flip is held for ~1 s, so
//   the pixel domain cannot miss it even at 12.5 MHz.
//
//   UPPERCASE ONLY: font_rom is indexed by the raw ASCII code (rom[ch*8+row])
//   and only holds NCH=91 entries, i.e. codes 0x00..0x5A.  Lowercase letters
//   (0x61+) address past the end of the ROM and read garbage, which is why the
//   first version drew nothing at all while an uppercase "TEST" test pattern
//   rendered fine.
//
//   silence (0) and unknown (1) are intentionally NOT displayed.
// =============================================================================
`timescale 1ns / 1ps

module word_to_text #(
    parameter integer HOLD_CYC = 12_500_000,   // 1 s at 12.5 MHz
    parameter integer NC       = 6
) (
    input  wire            clk,          // pixel clock
    input  wire            rst_n,
    input  wire            tog,          // toggles once per classification
    input  wire [3:0]      class_i,      // valid while `tog` is stable
    output wire            show,
    output wire [NC*8-1:0] text
);

    // ---- capture a new classification --------------------------------------
    reg [2:0] sys_s;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) sys_s <= 3'b000;
        else        sys_s <= {sys_s[1:0], tog};

    wire new_class = sys_s[2] ^ sys_s[1];      // any flip

    reg [27:0] hold;
    reg [3:0]  cls_l;
    reg [NC*8-1:0] txt_l;

    wire tick = (hold == 28'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hold  <= 28'd0;
            cls_l <= 4'd0;
            txt_l <= {NC*8{1'b0}};
        end else if (new_class) begin
            cls_l <= class_i;
            hold  <= HOLD_CYC[27:0];
            case (class_i)
                4'd2:  txt_l <= " YES ";
                4'd3:  txt_l <= " NO  ";
                4'd4:  txt_l <= " UP  ";
                4'd5:  txt_l <= "DOWN ";
                4'd6:  txt_l <= "LEFT ";
                4'd7:  txt_l <= "RIGHT";
                4'd8:  txt_l <= " ON  ";
                4'd9:  txt_l <= " OFF ";
                4'd10: txt_l <= "STOP ";
                4'd11: txt_l <= " GO  ";
                default: begin txt_l <= {NC*8{1'b0}}; hold <= 28'd0; end
            endcase
        end else if (!tick) begin
            hold <= hold - 28'd1;
        end
    end

    assign show = !tick;
    assign text = txt_l;

endmodule
