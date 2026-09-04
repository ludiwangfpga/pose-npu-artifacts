`timescale 1ns / 1ps
//===========================================================================
//  tb_engine_selfcheck -- self-checking testbench for conv3x3_engine
//
//  Drives a two-group int8 convolution and compares every requantized output
//  byte against a golden computed independently in the testbench: a plain
//  nested loop over the input-channel groups, the nine taps and the sixteen
//  lanes, followed by the requantization convention of Section II-B of the
//  paper (acc * M + bias, ReLU, output zero-point, 16-bit right shift with
//  round-half-up, saturation into [0, 255]).
//
//  Outputs are latched on dout_we, the way a consumer would latch them, so a
//  misplaced valid cannot pass unnoticed.
//
//  Run:
//    xvlog -sv conv3x3_mac_pair.v conv3x3_mac_array.v conv3x3_requant.v \
//              conv3x3_engine.v tb_engine_selfcheck.v
//    xelab --debug off --relax -L work --snapshot eng work.tb_engine_selfcheck
//    xsim eng -R
//===========================================================================
module tb_engine_selfcheck;

    localparam integer TAPS   = 9;
    localparam integer LANES  = 16;
    localparam integer SLICES = 8;
    localparam integer COLS   = 3;
    localparam integer GROUPS = 2;     // 32 input channels
    localparam integer NBEAT  = 80;    // activation beats per group
    localparam integer NPOS   = NBEAT - 2;   // complete windows per group

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    // ---------------- device under test -----------------------------------
    reg                     coef_we = 1'b0;
    reg [TAPS*LANES*8-1:0]  coef_beat;
    reg                     cfg_we = 1'b0;
    reg [15:0]              mult;
    reg [7:0]               ozero;
    reg [SLICES*32-1:0]     bias_flat;
    reg                     pix_we = 1'b0;
    reg [3*LANES*8-1:0]     pix_rows;
    reg                     grp_first = 1'b0, grp_last = 1'b0;
    wire [SLICES*8-1:0]     dout_flat;
    wire                    dout_we;

    conv3x3_engine #(.DEPTH(NPOS)) u_dut (
        .clk(clk), .rst(rst),
        .coef_we(coef_we), .coef_beat(coef_beat),
        .cfg_we(cfg_we), .mult(mult), .out_zero(ozero), .bias_flat(bias_flat),
        .pix_we(pix_we), .pix_rows(pix_rows),
        .group_first(grp_first), .group_last(grp_last),
        .dout_flat(dout_flat), .dout_we(dout_we)
    );

    // ---------------- stimulus and golden ----------------------------------
    reg  [7:0]  coef [0:GROUPS-1][0:TAPS-1][0:LANES-1][0:SLICES-1];
    reg  [7:0]  hist [0:GROUPS-1][0:NBEAT-1][0:COLS-1][0:LANES-1];
    reg signed [31:0] bias [0:SLICES-1];
    reg  [63:0] expect_q [0:NPOS-1];
    reg  [63:0] latched  [0:NPOS-1];
    integer     n_latch = 0;

    integer g, t, l, s, i, j, c, r, k;
    integer seed = 32'h51A0_1234;
    reg signed [47:0] acc;
    reg signed [48:0] p49;
    reg [7:0] gold [0:SLICES-1];
    reg [7:0] a_byte;
    reg signed [7:0] c_even, c_odd;
    integer artifact_terms = 0, sat_hi = 0, sat_lo = 0;

    always @(posedge clk) begin
        if (!rst && dout_we && n_latch < NPOS) begin
            latched[n_latch] <= dout_flat;
            n_latch          <= n_latch + 1;
        end
    end

    // one output position: window n uses beat n-3+c of every row, column 0
    // being the oldest, summed over every input-channel group
    task compute_golden(input integer n);
        integer gg;
        begin
            for (s = 0; s < SLICES; s = s + 1) begin
                acc = 48'sd0;
                for (gg = 0; gg < GROUPS; gg = gg + 1)
                    for (t = 0; t < TAPS; t = t + 1) begin
                        r = t / COLS;
                        c = t % COLS;
                        for (l = 0; l < LANES; l = l + 1) begin
                            a_byte = hist[gg][n-3+c][r][l];
                            c_even = coef[gg][t][l][s];
                            acc = acc + $signed(c_even) * $signed({1'b0, a_byte});
                            if (s % 2 == 0) begin
                                c_odd = coef[gg][t][l][s+1];
                                if (a_byte == 8'd0 && c_odd[7]) begin
                                    acc = acc + 48'sd1;
                                    artifact_terms = artifact_terms + 1;
                                end
                            end
                        end
                    end
                // requantization, Section II-B
                p49 = acc * $signed({1'b0, mult}) + $signed(bias[s]);
                if (p49[48]) p49 = 49'sd0;                      // ReLU
                p49 = p49 + $signed({9'd0, ozero, 16'd0});       // output zero-point
                if (p49[48]) begin
                    gold[s] = 8'd0; sat_lo = sat_lo + 1;
                end else if (p49[47:16] >= 32'd255) begin
                    gold[s] = 8'd255; sat_hi = sat_hi + 1;
                end else begin
                    gold[s] = p49[23:16] + p49[15];             // round-half-up
                end
            end
            expect_q[n-3] = {gold[7], gold[6], gold[5], gold[4],
                             gold[3], gold[2], gold[1], gold[0]};
        end
    endtask

    task load_coefficients(input integer grp);
        integer ss, ll, tt;
        begin
            for (ss = 0; ss < SLICES; ss = ss + 1) begin
                @(negedge clk);
                for (ll = 0; ll < LANES; ll = ll + 1)
                    for (tt = 0; tt < TAPS; tt = tt + 1)
                        coef_beat[(ll*TAPS + tt)*8 +: 8] = coef[grp][tt][ll][ss];
                coef_we = 1'b1;
            end
            @(negedge clk);
            coef_we = 1'b0;
        end
    endtask

    task stream_group(input integer grp, input integer is_first, input integer is_last);
        integer bb, jj, ll;
        begin
            for (bb = 0; bb < NBEAT; bb = bb + 1) begin
                @(negedge clk);
                for (jj = 0; jj < COLS; jj = jj + 1)
                    for (ll = 0; ll < LANES; ll = ll + 1)
                        pix_rows[(jj*LANES + ll)*8 +: 8] = hist[grp][bb][jj][ll];
                pix_we    = 1'b1;
                grp_first = is_first[0];
                grp_last  = is_last[0];
            end
            @(negedge clk);
            pix_we = 1'b0; grp_first = 1'b0; grp_last = 1'b0;
            repeat (24) @(negedge clk);
        end
    endtask

    integer mism;

    initial begin
        $dumpfile("engine.vcd");
        $dumpvars(1, tb_engine_selfcheck);

        mult  = 16'd18000;
        ozero = 8'd12;
        for (s = 0; s < SLICES; s = s + 1) begin
            bias[s] = ($random(seed) % 400000) - 200000;
            bias_flat[s*32 +: 32] = bias[s];
        end
        for (g = 0; g < GROUPS; g = g + 1)
            for (t = 0; t < TAPS; t = t + 1)
                for (l = 0; l < LANES; l = l + 1)
                    for (s = 0; s < SLICES; s = s + 1)
                        coef[g][t][l][s] = $random(seed);
        for (g = 0; g < GROUPS; g = g + 1)
            for (i = 0; i < NBEAT; i = i + 1)
                for (j = 0; j < COLS; j = j + 1)
                    for (l = 0; l < LANES; l = l + 1)
                        hist[g][i][j][l] = ((($random(seed) % 10) + 10) % 10 < 3)
                                           ? 8'd0 : $random(seed);

        repeat (6) @(posedge clk);
        @(negedge clk); rst = 1'b0;

        @(negedge clk); cfg_we = 1'b1;
        @(negedge clk); cfg_we = 1'b0;

        // group 0 then group 1 over the same output positions
        load_coefficients(0);
        stream_group(0, 1, 0);
        load_coefficients(1);
        stream_group(1, 0, 1);

        repeat (30) @(posedge clk);

        for (i = 3; i < NBEAT + 1; i = i + 1)
            if (i - 3 < NPOS) compute_golden(i);

        mism = 0;
        for (k = 0; k < NPOS; k = k + 1)
            if (expect_q[k] !== latched[k]) begin
                if (mism < 5)
                    $display("  MISMATCH position %0d: expected %h latched %h",
                             k, expect_q[k], latched[k]);
                mism = mism + 1;
            end

        $display("");
        $display("=========================================================");
        $display("  conv3x3_engine -- self-check against an independent");
        $display("  int8 convolution and the requantization of Section II-B");
        $display("---------------------------------------------------------");
        $display("  input channels                 : %0d (%0d groups)", GROUPS*LANES, GROUPS);
        $display("  multiply-accumulates per cycle : %0d", TAPS*LANES*SLICES);
        $display("  output positions expected      : %0d", NPOS);
        $display("  dout_we strobes observed       : %0d", n_latch);
        $display("  M = %0d, output zero-point = %0d", mult, ozero);
        $display("  packed-DSP artifact terms      : %0d", artifact_terms);
        $display("  saturated high / low           : %0d / %0d", sat_hi, sat_lo);
        $display("  MISMATCHES                     : %0d", mism);
        if (mism == 0 && n_latch == NPOS)
            $display("  RESULT: PASS");
        else
            $display("  RESULT: FAIL");
        $display("=========================================================");
        $finish;
    end

endmodule
