`timescale 1ns/1ps
module tb_word;
  reg clk=0, rst_n=0; always #40 clk=~clk;      // 12.5 MHz
  reg tog=0; reg [3:0] cls=0;
  wire show; wire [47:0] text;
  word_to_text #(.HOLD_CYC(2000)) dut(.clk(clk),.rst_n(rst_n),.tog(tog),.class_i(cls),.show(show),.text(text));
  integer i, shown=0;
  // fire a classification for class 11 ("go") the way board_top does
  task fire(input [3:0] c); begin
    @(negedge clk); cls=c; tog=~tog;
  end endtask
  initial begin
    repeat(5) @(posedge clk); rst_n=1;
    repeat(5) @(posedge clk);
    if (show) begin $display("FAIL: show should be low before any result"); $finish; end
    fire(11);
    repeat(4) @(posedge clk);
    if (!show) begin $display("FAIL: show never asserted"); $finish; end
    $display("text = \"%s\" (len %0d)", text, 6);
    if (text !== " GO  ") begin $display("FAIL: wrong text"); $finish; end
    $display("show=1 after result  OK");
    // hold must expire
    for (i=0;i<3000;i=i+1) begin @(posedge clk); if (show) shown=shown+1; end
    $display("show was high for ~%0d cycles (HOLD_CYC=2000)", shown);
    if (show) begin $display("FAIL: show did not expire"); $finish; end
    // silence must NOT display
    fire(0); repeat(10) @(posedge clk);
    if (show) begin $display("FAIL: silence displayed"); $finish; end
    $display("WORD_TO_TEXT: PASS");
    $finish;
  end
endmodule
