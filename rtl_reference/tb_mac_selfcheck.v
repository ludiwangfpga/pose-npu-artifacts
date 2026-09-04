`timescale 1ns / 1ps
//===========================================================================
//  tb_mac_selfcheck -- self-checking testbench for conv3x3_mac_array
//
//  The expected partial sums are computed independently, with a plain nested
//  loop over the nine taps and the sixteen input channels, so the model is
//  checked against the arithmetic the paper states and not against another
//  copy of the same RTL.
//
//  Results are latched the way a consumer would latch them, on psum_we, so
//  the strobe has to be aligned with the data it labels; capturing every
//  cycle instead would hide a misplaced valid.
//
//  Stimulus phases:
//    A  random coefficients and random activations, about a third of the
//       activation bytes forced to zero
//    B  an all-zero activation word, which isolates the packed-multiplier
//       artifact: an even-lane output is then exactly the number of negative
//       odd-lane coefficients paired with it
//    C  the extremes, coefficient -128 and +127 against activations 0 and 255
//
//  All stimulus is applied on the falling edge, so nothing the array samples
//  changes in the same time step as the sampling edge.
//
//  Run:
//    xvlog -sv conv3x3_mac_pair.v conv3x3_mac_array.v tb_mac_selfcheck.v
//    xelab --debug off --relax -L work --snapshot macref work.tb_mac_selfcheck
//    xsim macref -R
//===========================================================================
module tb_mac_selfcheck;

    localparam integer TAPS   = 9;
    localparam integer LANES  = 16;
    localparam integer SLICES = 8;
    localparam integer COLS   = 3;
    localparam integer NDRIVE = 560;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    // ---------------- device under test ---------------------------------
    reg                     coef_we = 1'b0;
    reg  [TAPS*LANES*8-1:0] coef_beat;
    reg                     pix_we  = 1'b0;
    reg  [3*LANES*8-1:0]    pix_rows;

    wire signed [23:0] psum_0, psum_1, psum_2, psum_3;
    wire signed [23:0] psum_4, psum_5, psum_6, psum_7;
    wire               psum_we;

    conv3x3_mac_array u_dut (
        .clk(clk), .rst(rst),
        .coef_we(coef_we), .coef_beat(coef_beat),
        .pix_we(pix_we),   .pix_rows(pix_rows),
        .psum_0(psum_0), .psum_1(psum_1), .psum_2(psum_2), .psum_3(psum_3),
        .psum_4(psum_4), .psum_5(psum_5), .psum_6(psum_6), .psum_7(psum_7),
        .psum_we(psum_we)
    );

    // ---------------- stimulus and independent golden -------------------
    reg  [7:0]   coef     [0:TAPS-1][0:LANES-1][0:SLICES-1];
    reg  [7:0]   hist     [0:NDRIVE-1][0:COLS-1][0:LANES-1];
    reg  [191:0] expect_q [0:NDRIVE-1];
    reg  [191:0] latched  [0:NDRIVE-1];
    integer      n_expect = 0;
    integer      n_latch  = 0;

    integer t, l, s, i, j, c, r;
    integer seed = 32'h0BADC0DE;
    reg signed [23:0] acc;
    reg signed [23:0] gold [0:SLICES-1];
    reg        [7:0]  a_byte;
    reg signed [7:0]  c_even, c_odd;
    integer zero_bytes = 0, artifact_terms = 0;

    // latch exactly as a consumer would: on the strobe
    always @(posedge clk) begin
        if (!rst && psum_we && n_latch < NDRIVE) begin
            latched[n_latch] <= {psum_7, psum_6, psum_5, psum_4,
                                 psum_3, psum_2, psum_1, psum_0};
            n_latch          <= n_latch + 1;
        end
    end

    // window n takes column c of every row from beat n-3+c, column 0 oldest
    task compute_golden(input integer n);
        begin
            for (s = 0; s < SLICES; s = s + 1) begin
                acc = 24'sd0;
                for (t = 0; t < TAPS; t = t + 1) begin
                    r = t / COLS;
                    c = t % COLS;
                    for (l = 0; l < LANES; l = l + 1) begin
                        a_byte = hist[n-3+c][r][l];
                        c_even = coef[t][l][s];
                        acc = acc + $signed(c_even) * $signed({1'b0, a_byte});
                        if (s % 2 == 0) begin
                            c_odd = coef[t][l][s+1];
                            if (a_byte == 8'd0 && c_odd[7]) begin
                                acc = acc + 24'sd1;
                                artifact_terms = artifact_terms + 1;
                            end
                        end
                    end
                end
                gold[s] = acc;
            end
            expect_q[n_expect] = {gold[7], gold[6], gold[5], gold[4],
                                  gold[3], gold[2], gold[1], gold[0]};
            n_expect = n_expect + 1;
        end
    endtask

    task load_coefficients;
        integer ss, ll, tt;
        begin
            for (ss = 0; ss < SLICES; ss = ss + 1) begin
                @(negedge clk);
                for (ll = 0; ll < LANES; ll = ll + 1)
                    for (tt = 0; tt < TAPS; tt = tt + 1)
                        coef_beat[(ll*TAPS + tt)*8 +: 8] = coef[tt][ll][ss];
                coef_we = 1'b1;
            end
            @(negedge clk);
            coef_we = 1'b0;
        end
    endtask

    // ---------------- main ----------------------------------------------
    integer mismatches, compared, k;

    initial begin
        // the interface signals only, so the dump stays small enough to ship
        $dumpfile("mac_array.vcd");
        $dumpvars(1, tb_mac_selfcheck);

        for (t = 0; t < TAPS; t = t + 1)
            for (l = 0; l < LANES; l = l + 1)
                for (s = 0; s < SLICES; s = s + 1)
                    coef[t][l][s] = $random(seed);
        // phase C pins the coefficient extremes on the first four lanes
        for (t = 0; t < TAPS; t = t + 1)
            for (l = 0; l < 4; l = l + 1) begin
                coef[t][l][0] =  8'sd127;
                coef[t][l][1] = -8'sd128;
                coef[t][l][2] = -8'sd128;
                coef[t][l][3] =  8'sd127;
            end

        for (i = 0; i < NDRIVE; i = i + 1)
            for (j = 0; j < COLS; j = j + 1)
                for (l = 0; l < LANES; l = l + 1) begin
                    if (i >= 200 && i < 240)
                        hist[i][j][l] = 8'd0;
                    else if (i >= 300 && i < 340)
                        hist[i][j][l] = (l % 2) ? 8'd255 : 8'd0;
                    else if ((($random(seed) % 10) + 10) % 10 < 3)
                        hist[i][j][l] = 8'd0;
                    else
                        hist[i][j][l] = $random(seed);
                    if (hist[i][j][l] == 8'd0) zero_bytes = zero_bytes + 1;
                end

        repeat (6) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        load_coefficients;
        repeat (3) @(negedge clk);

        for (i = 0; i < NDRIVE; i = i + 1) begin
            @(negedge clk);
            for (j = 0; j < COLS; j = j + 1)
                for (l = 0; l < LANES; l = l + 1)
                    pix_rows[(j*LANES + l)*8 +: 8] = hist[i][j][l];
            pix_we = 1'b1;
            if (i >= 3) compute_golden(i);
        end
        @(negedge clk);
        pix_we = 1'b0;
        // the last window closes on the final beat, one beyond the loop
        compute_golden(NDRIVE);
        repeat (30) @(posedge clk);

        // the strobe must label the sample it accompanies, so the latched
        // sequence is compared position by position with no realignment
        mismatches = 0;
        compared = (n_latch < n_expect) ? n_latch : n_expect;
        for (k = 0; k < compared; k = k + 1)
            if (expect_q[k] !== latched[k]) begin
                if (mismatches < 5)
                    $display("  MISMATCH at strobe %0d: expected %h latched %h",
                             k, expect_q[k], latched[k]);
                mismatches = mismatches + 1;
            end

        $display("");
        $display("=========================================================");
        $display("  conv3x3_mac_array -- self-check against an independent");
        $display("  nested-loop model of the arithmetic stated in the paper");
        $display("---------------------------------------------------------");
        $display("  multiply-accumulates per cycle : %0d", TAPS*LANES*SLICES);
        $display("  activation beats driven        : %0d", NDRIVE);
        $display("  complete windows expected      : %0d", n_expect);
        $display("  psum_we strobes observed       : %0d", n_latch);
        $display("  samples compared               : %0d", compared);
        $display("  zero activation bytes driven   : %0d", zero_bytes);
        $display("  packed-DSP artifact terms      : %0d", artifact_terms);
        $display("  MISMATCHES                     : %0d", mismatches);
        if (mismatches == 0 && n_latch == n_expect && compared > 400)
            $display("  RESULT: PASS");
        else
            $display("  RESULT: FAIL");
        $display("=========================================================");
        $finish;
    end

endmodule
