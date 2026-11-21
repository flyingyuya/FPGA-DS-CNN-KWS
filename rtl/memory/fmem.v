// =============================================================================
// fmem.v -- MFCC input memory: 4096 x 8 bit with a stream loader.
//
// The host (or a future on-chip MFCC front end) pushes one UINT8 code per cycle
// with a valid handshake; the stem reads them back through a registered port.  The
// layout is the plain raster order of features.py: code(row = MFCC coefficient,
// col = frame) lives at row*W_IN + col.
//
// The stem is the only layer that touches this memory, and it is also the only
// layer with a non-zero input zero point, so out-of-range reads must be replaced
// by ZP_IN rather than 0.  That substitution happens in conv_engine's byte path;
// this module only has to store bytes, which is why it needs no reserved word and
// no power-on zeroing.
//
// NOTE on block RAM inference: the memory write lives in its own clocked block
// with no reset in the sensitivity list.  Folding it into the loader FSM's
// asynchronous-reset block made Vivado give up on BRAM and build 33k flip-flops
// (13k LUTs) instead -- a 4 KB buffer must never cost that.
// =============================================================================
`timescale 1ns / 1ps

module fmem #(
    parameter integer DEPTH    = 4096,
    parameter integer ADDR_W   = 12,
    parameter integer ACT_BITS = 8,
    parameter         INIT_FILE = ""
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- stream load -------------------------------------------------------
    input  wire                    load_start,
    input  wire [15:0]             load_len,     // bytes to accept
    input  wire                    s_valid,
    input  wire [ACT_BITS-1:0]     s_data,
    input  wire                    s_last,
    output wire                    s_ready,
    output reg                     load_busy,
    output reg                     load_done,

    // ---- direct map write (MFCC front end) --------------------------------
    input  wire                    ext_we,
    input  wire [ADDR_W-1:0]       ext_addr,
    input  wire [ACT_BITS-1:0]     ext_data,

    // ---- read port (stem) --------------------------------------------------
    input  wire                    re,
    input  wire [ADDR_W-1:0]       raddr,
    output reg  [ACT_BITS-1:0]     rdata
);

    (* ram_style = "block" *)
    reg [ACT_BITS-1:0] mem [0:DEPTH-1];

    reg [ADDR_W-1:0]   wr;
    reg                mem_we;

    assign s_ready = load_busy;

    // simple dual-port memory: write at `wr`, read at `raddr`
    // ONE write port with muxed address/data.  Writing it as
    //   if (mem_we) mem[wr] <= s_data; else if (ext_we) mem[ext_addr] <= ...
    // looks equivalent but Vivado counts two write ports, cannot map them onto a
    // simple dual-port BRAM, and silently falls back to LUTRAM.
    wire                we = mem_we | ext_we;
    wire [ADDR_W-1:0]   wa = mem_we ? wr : ext_addr;
    wire [ACT_BITS-1:0] wd = mem_we ? s_data : ext_data;

    always @(posedge clk) begin
        if (we) mem[wa] <= wd;
        if (re) rdata   <= mem[raddr];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            load_busy <= 1'b0;
            load_done <= 1'b0;
            mem_we    <= 1'b0;
            wr        <= {ADDR_W{1'b0}};
        end else begin
            load_done <= 1'b0;
            mem_we    <= 1'b0;
            if (!load_busy) begin
                if (load_start) begin
                    load_busy <= 1'b1;
                    wr        <= {ADDR_W{1'b0}};
                end
            end else if (s_valid) begin
                mem_we <= 1'b1;
                wr     <= wr + 1'b1;
                if (s_last || (wr + 1'b1 == load_len[ADDR_W-1:0])) begin
                    load_busy <= 1'b0;
                    load_done <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
    end

endmodule
