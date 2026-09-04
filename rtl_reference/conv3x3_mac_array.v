`timescale 1ns / 1ps
//===========================================================================
//  conv3x3_mac_array
//
//  Reference model of the 3x3 convolution multiply-accumulate array of
//
//      "An Ethernet Instruction-Driven INT8 FPGA NPU for Single-Stage
//       Multi-Person Pose Estimation"          (Section IV-A, Fig. 5)
//
//  Every cycle the array performs
//
//      TAPS x LANES x SLICES = 9 x 16 x 8 = 1152
//
//  int8 multiply-accumulates: the nine taps of a 3x3 kernel, sixteen input
//  channels -- one NHWC-16 word -- and eight output channels, all unrolled in
//  the same cycle.  Feature-map rows and columns are not unrolled.  The three
//  rows of the window arrive together, one NHWC-16 word each; the three
//  columns are formed inside from three consecutive beats of that stream.
//
//  The reduction runs across the taps first and across the input channels
//  afterwards.  Integer addition is associative and the accumulator here is
//  wide enough to be lossless, so how the summation is split into pipeline
//  stages is an implementation choice and not a numerical one; this model
//  sums the nine taps of a lane in one stage and then folds the sixteen lane
//  sums with a balanced binary tree.
//
//  Accumulator width.  One product is bounded by 128*255 = 32640; the array
//  sums TAPS*LANES = 144 of them and can add up to 144 packed-DSP correction
//  terms, so |result| <= 4,700,304 < 2^23 and a flat 24-bit signed
//  accumulator is lossless at every stage of this model.  Accumulation across
//  several NHWC-16 words, which the deployed engine performs downstream, is
//  outside this model.
//
//  Interface contract.
//    * A coefficient burst is exactly SLICES contiguous beats, one per output
//      channel.  A gap in coef_we restarts the burst at output channel 0.
//    * Coefficients must not be reloaded while an activation stream is in
//      flight: this model holds one coefficient set, not a shadow buffer.
//    * row_delay advances only while pix_we is high, so a gap in the
//      activation stream stalls the window instead of corrupting it.
//    * psum_we marks a sample whose three-column window lies entirely inside
//      the stream; the first two samples after a stream starts are masked.
//
//  Scope.  This file models the arithmetic and the dataflow that the paper
//  describes for the 3x3 engine.  Line buffering, DDR-side addressing,
//  coefficient double buffering, requantization, and the 1x1 engine that
//  shares the same packed-multiplier scheme are not part of this model.
//  Its self-check is tb_mac_selfcheck.v and the captured run is in sim/.
//===========================================================================
module conv3x3_mac_array #(
    parameter integer TAPS    = 9,     // 3x3 kernel, tap index = row*3 + col
    parameter integer LANES   = 16,    // input channels per NHWC-16 word
    parameter integer SLICES  = 8,     // output channels computed together
    parameter integer ACC_W   = 24,    // lossless accumulator width
    parameter integer MUL_LAT = 5      // pipeline depth of one multiplier pair
)(
    input  wire clk,
    input  wire rst,

    // ---- coefficient load: one beat per output channel, SLICES beats -----
    // Beat layout is lane-major: byte (lane*TAPS + tap) is the coefficient of
    // that tap and that input channel for the output channel being loaded.
    input  wire                        coef_we,
    input  wire [TAPS*LANES*8-1:0]     coef_beat,

    // ---- activation stream: the three rows of the window, one word each --
    // Row-major: byte (row*LANES + lane).
    input  wire                        pix_we,
    input  wire [3*LANES*8-1:0]        pix_rows,

    // ---- partial sums, one per output channel ---------------------------
    output wire signed [ACC_W-1:0]     psum_0,
    output wire signed [ACC_W-1:0]     psum_1,
    output wire signed [ACC_W-1:0]     psum_2,
    output wire signed [ACC_W-1:0]     psum_3,
    output wire signed [ACC_W-1:0]     psum_4,
    output wire signed [ACC_W-1:0]     psum_5,
    output wire signed [ACC_W-1:0]     psum_6,
    output wire signed [ACC_W-1:0]     psum_7,
    output wire                        psum_we
);

    localparam integer COLS       = 3;
    localparam integer TREE_DEPTH = 4;      // log2(LANES)

    // registered stages from pix_rows to the outputs:
    //   window register 1 + multiplier pair MUL_LAT + tap sum 1 + tree stages
    localparam integer LATENCY = 1 + MUL_LAT + 1 + TREE_DEPTH;

    genvar t, l, s, p, n;

    // =====================================================================
    //  coefficient store
    //  One byte per (slice, lane, tap).  The beat counter restarts at zero
    //  whenever the burst is not active, so a burst always begins at output
    //  channel 0.
    // =====================================================================
    reg [7:0]              coef_mem [0:SLICES-1][0:LANES-1][0:TAPS-1];
    reg [$clog2(SLICES):0] beat_idx;

    integer li, ti;
    always @(posedge clk) begin
        if (rst || !coef_we) begin
            beat_idx <= 0;
        end else begin
            for (li = 0; li < LANES; li = li + 1)
                for (ti = 0; ti < TAPS; ti = ti + 1)
                    coef_mem[beat_idx][li][ti] <=
                        coef_beat[(li*TAPS + ti)*8 +: 8];
            beat_idx <= (beat_idx == SLICES-1) ? 0 : beat_idx + 1;
        end
    end

    // =====================================================================
    //  window formation
    //  The three rows arrive together; the three columns are the newest three
    //  beats of that stream, column 0 being the oldest.
    // =====================================================================
    reg [LANES*8-1:0] row_delay [0:COLS-1][0:COLS-1];     // [row][age]

    integer ri, ai;
    always @(posedge clk) begin
        if (rst) begin
            for (ri = 0; ri < COLS; ri = ri + 1)
                for (ai = 0; ai < COLS; ai = ai + 1)
                    row_delay[ri][ai] <= {(LANES*8){1'b0}};
        end else if (pix_we) begin
            for (ri = 0; ri < COLS; ri = ri + 1) begin
                for (ai = COLS-1; ai > 0; ai = ai - 1)
                    row_delay[ri][ai] <= row_delay[ri][ai-1];
                row_delay[ri][0] <= pix_rows[ri*LANES*8 +: LANES*8];
            end
        end
    end

    // tap t = row*3 + col, and column 0 is the oldest beat, i.e. age COLS-1-col
    wire [7:0] act_of [0:TAPS-1][0:LANES-1];
    generate
        for (t = 0; t < TAPS; t = t + 1) begin : TAP_ACT
            localparam integer TR = t / COLS;
            localparam integer TC = t % COLS;
            for (l = 0; l < LANES; l = l + 1) begin : TAP_ACT_LANE
                assign act_of[t][l] = row_delay[TR][COLS-1-TC][l*8 +: 8];
            end
        end
    endgenerate

    // =====================================================================
    //  the 1152 multipliers
    //  Output channels are taken two at a time; the pair shares its
    //  activation operand.
    // =====================================================================
    wire signed [15:0] prod [0:TAPS-1][0:LANES-1][0:SLICES-1];

    generate
        for (t = 0; t < TAPS; t = t + 1) begin : MUL_TAP
            for (l = 0; l < LANES; l = l + 1) begin : MUL_LANE
                for (p = 0; p < SLICES/2; p = p + 1) begin : MUL_PAIR
                    conv3x3_mac_pair #(.LATENCY(MUL_LAT)) u_pair (
                        .clk       (clk),
                        .rst       (rst),
                        .coef_even (coef_mem[p*2  ][l][t]),
                        .coef_odd  (coef_mem[p*2+1][l][t]),
                        .act       (act_of[t][l]),
                        .prod_even (prod[t][l][p*2  ]),
                        .prod_odd  (prod[t][l][p*2+1])
                    );
                end
            end
        end
    endgenerate

    // =====================================================================
    //  reduction, taps first
    // =====================================================================
    reg signed [ACC_W-1:0] lane_sum [0:SLICES-1][0:LANES-1];

    generate
        for (s = 0; s < SLICES; s = s + 1) begin : TAPSUM_SLICE
            for (l = 0; l < LANES; l = l + 1) begin : TAPSUM_LANE
                integer k;
                reg signed [ACC_W-1:0] acc;
                always @(posedge clk) begin
                    if (rst) begin
                        lane_sum[s][l] <= {ACC_W{1'b0}};
                    end else begin
                        acc = {ACC_W{1'b0}};
                        for (k = 0; k < TAPS; k = k + 1)
                            acc = acc + $signed(prod[k][l][s]);
                        lane_sum[s][l] <= acc;
                    end
                end
            end
        end
    endgenerate

    // =====================================================================
    //  reduction, input channels afterwards: a balanced binary tree
    // =====================================================================
    reg signed [ACC_W-1:0] tree [0:TREE_DEPTH-1][0:SLICES-1][0:LANES/2-1];

    generate
        for (s = 0; s < SLICES; s = s + 1) begin : TREE_SLICE
            for (t = 0; t < TREE_DEPTH; t = t + 1) begin : TREE_STAGE
                localparam integer NODES = LANES >> (t + 1);
                for (n = 0; n < NODES; n = n + 1) begin : TREE_NODE
                    if (t == 0) begin : FROM_LANES
                        always @(posedge clk)
                            if (rst) tree[0][s][n] <= {ACC_W{1'b0}};
                            else     tree[0][s][n] <= $signed(lane_sum[s][n*2])
                                                    + $signed(lane_sum[s][n*2+1]);
                    end else begin : FROM_TREE
                        always @(posedge clk)
                            if (rst) tree[t][s][n] <= {ACC_W{1'b0}};
                            else     tree[t][s][n] <= $signed(tree[t-1][s][n*2])
                                                    + $signed(tree[t-1][s][n*2+1]);
                    end
                end
            end
        end
    endgenerate

    // =====================================================================
    //  output valid
    //  The array is a fixed-latency pipeline, so the strobe is the accepted
    //  activation beat delayed by LATENCY.  A window spans three consecutive
    //  beats, so a sample counts only if the beat two positions earlier was
    //  also accepted; that masks the two head samples of every stream.
    // =====================================================================
    reg [LATENCY+1:0] we_pipe;
    always @(posedge clk) begin
        if (rst) we_pipe <= {(LATENCY+2){1'b0}};
        else     we_pipe <= {we_pipe[LATENCY:0], pix_we};
    end

    assign psum_0  = tree[TREE_DEPTH-1][0][0];
    assign psum_1  = tree[TREE_DEPTH-1][1][0];
    assign psum_2  = tree[TREE_DEPTH-1][2][0];
    assign psum_3  = tree[TREE_DEPTH-1][3][0];
    assign psum_4  = tree[TREE_DEPTH-1][4][0];
    assign psum_5  = tree[TREE_DEPTH-1][5][0];
    assign psum_6  = tree[TREE_DEPTH-1][6][0];
    assign psum_7  = tree[TREE_DEPTH-1][7][0];
    assign psum_we = we_pipe[LATENCY-1] & we_pipe[LATENCY+1];

endmodule
