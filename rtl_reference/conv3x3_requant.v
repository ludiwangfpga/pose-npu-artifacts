`timescale 1ns / 1ps
//===========================================================================
//  conv3x3_requant
//
//  The output stage of the 3x3 convolution engine: it accumulates the partial
//  sums of the input-channel groups and turns the completed accumulator into
//  an unsigned int8 activation, following the requantization convention that
//  Section II-B of
//
//      "An Ethernet Instruction-Driven INT8 FPGA NPU for Single-Stage
//       Multi-Person Pose Estimation"
//
//  states for the convolution path: a per-layer multiplier
//  M = round(s_in * s_w / s_out * 2^16), a 16-bit right shift with round-half-up
//  carry, and a bias folded into int32.  Activations are unsigned, so the
//  output zero-point is added before the shift and the result is saturated
//  into [0, 255].  ReLU is applied in the scaled domain, before the zero-point,
//  which is what makes a clamp at zero equivalent to the activation function.
//
//  Order of operations, per output channel:
//
//      acc   = sum over input-channel groups of the array's partial sums
//      p     = acc * M + bias                      (49-bit signed)
//      p     = (p < 0) ? 0 : p                     (ReLU)
//      p     = p + (out_zero << 16)                (output zero-point)
//      dout  = saturate_u8( (p + 2^15) >> 16 )     (round-half-up)
//
//  Reference model.  It states the arithmetic the paper describes; it is not
//  the deployed source, and the deployed engine's scratchpad organisation,
//  bias double buffering and stride handling are not modelled here.
//===========================================================================
module conv3x3_requant #(
    parameter integer SLICES = 8,      // output channels handled together
    parameter integer ACC_W  = 24,     // width of one partial sum
    parameter integer SUM_W  = 32,     // width of the group accumulator
    parameter integer PROD_W = 49,     // width of accumulator x multiplier
    parameter integer DEPTH  = 320     // output positions held per group pass
)(
    input  wire clk,
    input  wire rst,

    // ---- per-layer constants, latched on cfg_we -------------------------
    input  wire                        cfg_we,
    input  wire [15:0]                 mult,        // M
    input  wire [7:0]                  out_zero,    // output zero-point
    input  wire [SLICES*32-1:0]        bias_flat,   // int32 per output channel

    // ---- partial sums from the MAC array --------------------------------
    // group_first clears the accumulator, group_last releases the output
    input  wire                        group_first,
    input  wire                        group_last,
    input  wire                        psum_we,
    input  wire [SLICES*ACC_W-1:0]     psum_flat,

    // ---- requantized activations ----------------------------------------
    output wire [SLICES*8-1:0]         dout_flat,
    output wire                        dout_we
);

    localparam integer PW = (DEPTH <= 2) ? 1 : $clog2(DEPTH);

    genvar s;
    integer si;

    // ---- latched constants ---------------------------------------------
    reg [15:0]  m_reg;
    reg [7:0]   z_reg;
    reg [31:0]  bias_reg [0:SLICES-1];

    always @(posedge clk) begin
        if (rst) begin
            m_reg <= 16'd0;
            z_reg <= 8'd0;
            for (si = 0; si < SLICES; si = si + 1)
                bias_reg[si] <= 32'sd0;
        end else if (cfg_we) begin
            m_reg <= mult;
            z_reg <= out_zero;
            for (si = 0; si < SLICES; si = si + 1)
                bias_reg[si] <= bias_flat[si*32 +: 32];
        end
    end

    // ---- output-position counter ----------------------------------------
    // one position per accepted partial sum; a gap restarts the pass
    reg [PW-1:0] pos;
    always @(posedge clk) begin
        if (rst || !psum_we) pos <= {PW{1'b0}};
        else                 pos <= (pos == DEPTH-1) ? {PW{1'b0}} : pos + 1'b1;
    end

    // ---- group accumulator ----------------------------------------------
    reg signed [SUM_W-1:0] acc_mem [0:DEPTH-1][0:SLICES-1];

    wire signed [SUM_W-1:0] running [0:SLICES-1];
    wire signed [SUM_W-1:0] widened [0:SLICES-1];

    generate
        for (s = 0; s < SLICES; s = s + 1) begin : ACCUM
            assign widened[s] = $signed(psum_flat[s*ACC_W +: ACC_W]);
            assign running[s] = (group_first ? {SUM_W{1'b0}} : acc_mem[pos][s])
                              + widened[s];
            always @(posedge clk)
                if (psum_we) acc_mem[pos][s] <= running[s];
        end
    endgenerate

    // ---- requantization pipeline ----------------------------------------
    // stage 1: scale and bias      stage 2: ReLU and zero-point
    // stage 3: round, saturate
    reg signed [PROD_W-1:0] scaled [0:SLICES-1];
    reg signed [PROD_W-1:0] shifted[0:SLICES-1];
    reg        [7:0]        result [0:SLICES-1];
    reg [2:0]               we_pipe;

    // round-half-up on a 16-bit binary point, saturated into a byte
    function automatic [7:0] round_saturate(input signed [PROD_W-1:0] v);
        begin
            if (v[PROD_W-1])                     round_saturate = 8'd0;
            else if (v[47:16] >= 32'd255)        round_saturate = 8'd255;
            else                                 round_saturate = v[23:16] + v[15];
        end
    endfunction

    generate
        for (s = 0; s < SLICES; s = s + 1) begin : REQUANT
            always @(posedge clk) begin
                if (rst) begin
                    scaled [s] <= {PROD_W{1'b0}};
                    shifted[s] <= {PROD_W{1'b0}};
                    result [s] <= 8'd0;
                end else begin
                    // stage 1
                    scaled[s] <= $signed(running[s]) * $signed({1'b0, m_reg})
                               + $signed(bias_reg[s]);
                    // stage 2: ReLU in the scaled domain, then the zero-point
                    shifted[s] <= (scaled[s][PROD_W-1] ? {PROD_W{1'b0}} : scaled[s])
                                + $signed({{(PROD_W-24){1'b0}}, z_reg, 16'd0});
                    // stage 3
                    result[s] <= round_saturate(shifted[s]);
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) we_pipe <= 3'd0;
        else     we_pipe <= {we_pipe[1:0], psum_we & group_last};
    end

    generate
        for (s = 0; s < SLICES; s = s + 1) begin : PACK
            assign dout_flat[s*8 +: 8] = result[s];
        end
    endgenerate

    assign dout_we = we_pipe[2];

endmodule
