`timescale 1ns / 1ps
//===========================================================================
//  conv3x3_mac_pair
//
//  Two int8 multiplications that share one activation operand.  This is the
//  pairing reported in Section V-B of
//
//      "An Ethernet Instruction-Driven INT8 FPGA NPU for Single-Stage
//       Multi-Person Pose Estimation"
//
//  The deployed engine folds such a pair onto a single DSP48E1, and a
//  sign-correction constant keeps the even-lane product exact whenever the
//  odd-lane product is negative.  A zero activation produces no borrow for
//  that constant to cancel, so the even-lane product then comes out one
//  larger.  That is the packed-DSP artifact the paper reports, and the model
//  below reproduces it rather than hiding it, because the compiler's golden
//  reproduces it too.
//
//  Reference model.  It states the arithmetic that the paper describes; it is
//  not the deployed source and does not model a vendor primitive.
//===========================================================================
module conv3x3_mac_pair #(
    parameter integer LATENCY = 5      // registered stages before the output
)(
    input  wire               clk,
    input  wire               rst,
    input  wire signed [7:0]  coef_even,
    input  wire signed [7:0]  coef_odd,
    input  wire        [7:0]  act,      // activations are unsigned bytes
    output wire signed [15:0] prod_even,
    output wire signed [15:0] prod_odd
);

    // an unsigned activation promoted to a positive signed operand
    wire signed [8:0] act_s = $signed({1'b0, act});

    // the artifact: it appears only when the shared operand is zero and the
    // odd-lane coefficient is negative
    wire correction = (act == 8'd0) & coef_odd[7];

    wire signed [17:0] wide_even = coef_even * act_s + $signed({17'd0, correction});
    wire signed [17:0] wide_odd  = coef_odd  * act_s;

    // |coef| <= 128 and act <= 255 bound both products to [-32640, 32385],
    // so sixteen bits carry them without loss.
    reg signed [15:0] delay_even [0:LATENCY-1];
    reg signed [15:0] delay_odd  [0:LATENCY-1];

    integer stage;
    always @(posedge clk) begin
        if (rst) begin
            for (stage = 0; stage < LATENCY; stage = stage + 1) begin
                delay_even[stage] <= 16'sd0;
                delay_odd [stage] <= 16'sd0;
            end
        end else begin
            delay_even[0] <= wide_even[15:0];
            delay_odd [0] <= wide_odd [15:0];
            for (stage = 1; stage < LATENCY; stage = stage + 1) begin
                delay_even[stage] <= delay_even[stage-1];
                delay_odd [stage] <= delay_odd [stage-1];
            end
        end
    end

    assign prod_even = delay_even[LATENCY-1];
    assign prod_odd  = delay_odd [LATENCY-1];

endmodule
