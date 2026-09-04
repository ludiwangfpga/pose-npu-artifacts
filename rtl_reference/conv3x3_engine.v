`timescale 1ns / 1ps
//===========================================================================
//  conv3x3_engine
//
//  The int8 3x3 convolution engine of
//
//      "An Ethernet Instruction-Driven INT8 FPGA NPU for Single-Stage
//       Multi-Person Pose Estimation"
//
//  composed from its two published parts: the multiply-accumulate array of
//  Section IV-A and Fig. 5, and the requantization convention of Section II-B.
//  One activation word in, one requantized activation word out:
//
//      conv3x3_mac_array   1152 int8 MACs per cycle, reduction across the
//                          nine taps first and the sixteen input channels
//                          second, 24-bit partial sums
//      conv3x3_requant     accumulation across input-channel groups, then
//                          acc * M + bias, ReLU, output zero-point, a 16-bit
//                          right shift with round-half-up, saturation to
//                          [0, 255]
//
//  A layer whose input has more than sixteen channels is computed as a
//  sequence of group passes over the same output positions: assert
//  group_first on the first pass and group_last on the last, and load the
//  coefficients of each group before its pass.
//
//  Reference model.  It states the arithmetic and the dataflow the paper
//  describes.  The line buffering that feeds the three rows, the DDR-side
//  addressing, the coefficient and bias double buffering, the stride-2 output
//  decimation and the instruction decoding of the deployed engine are not
//  modelled here, and the deployed source is not released.
//===========================================================================
module conv3x3_engine #(
    parameter integer TAPS    = 9,
    parameter integer LANES   = 16,
    parameter integer SLICES  = 8,
    parameter integer ACC_W   = 24,
    parameter integer MUL_LAT = 5,
    parameter integer DEPTH   = 320    // output positions held per group pass
)(
    input  wire clk,
    input  wire rst,

    // ---- coefficients of the group about to be streamed -----------------
    input  wire                        coef_we,
    input  wire [TAPS*LANES*8-1:0]     coef_beat,

    // ---- per-layer requantization constants -----------------------------
    input  wire                        cfg_we,
    input  wire [15:0]                 mult,
    input  wire [7:0]                  out_zero,
    input  wire [SLICES*32-1:0]        bias_flat,

    // ---- activation stream ----------------------------------------------
    input  wire                        pix_we,
    input  wire [3*LANES*8-1:0]        pix_rows,
    input  wire                        group_first,
    input  wire                        group_last,

    // ---- requantized activations -----------------------------------------
    output wire [SLICES*8-1:0]         dout_flat,
    output wire                        dout_we
);

    // mirrors the pipeline depth of conv3x3_mac_array: window register,
    // multiplier pair, tap sum, and one stage per level of the lane tree
    localparam integer ARRAY_LAT = 1 + MUL_LAT + 1 + 4;

    wire signed [ACC_W-1:0] psum [0:SLICES-1];
    wire                    psum_we;
    wire [SLICES*ACC_W-1:0] psum_flat;

    conv3x3_mac_array #(
        .TAPS(TAPS), .LANES(LANES), .SLICES(SLICES),
        .ACC_W(ACC_W), .MUL_LAT(MUL_LAT)
    ) u_array (
        .clk(clk), .rst(rst),
        .coef_we(coef_we), .coef_beat(coef_beat),
        .pix_we(pix_we),   .pix_rows(pix_rows),
        .psum_0(psum[0]), .psum_1(psum[1]), .psum_2(psum[2]), .psum_3(psum[3]),
        .psum_4(psum[4]), .psum_5(psum[5]), .psum_6(psum[6]), .psum_7(psum[7]),
        .psum_we(psum_we)
    );

    genvar s;
    generate
        for (s = 0; s < SLICES; s = s + 1) begin : FLATTEN
            assign psum_flat[s*ACC_W +: ACC_W] = psum[s];
        end
    endgenerate

    // the group markers travel with the activation beat, so they have to meet
    // the partial sums at the far end of the array
    reg [ARRAY_LAT-1:0] first_pipe, last_pipe;
    always @(posedge clk) begin
        if (rst) begin
            first_pipe <= {ARRAY_LAT{1'b0}};
            last_pipe  <= {ARRAY_LAT{1'b0}};
        end else begin
            first_pipe <= {first_pipe[ARRAY_LAT-2:0], group_first};
            last_pipe  <= {last_pipe [ARRAY_LAT-2:0], group_last};
        end
    end

    conv3x3_requant #(
        .SLICES(SLICES), .ACC_W(ACC_W), .DEPTH(DEPTH)
    ) u_requant (
        .clk(clk), .rst(rst),
        .cfg_we(cfg_we), .mult(mult), .out_zero(out_zero), .bias_flat(bias_flat),
        .group_first(first_pipe[ARRAY_LAT-1]),
        .group_last (last_pipe [ARRAY_LAT-1]),
        .psum_we(psum_we), .psum_flat(psum_flat),
        .dout_flat(dout_flat), .dout_we(dout_we)
    );

endmodule
