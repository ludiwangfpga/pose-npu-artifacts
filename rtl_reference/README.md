# Reference model of the 3x3 convolution engine

`conv3x3_mac_array.v` and `conv3x3_mac_pair.v` are a reference implementation of
the multiply-accumulate array the paper describes in Section IV-A and Fig. 5,
together with the packed-multiplier behaviour it reports in Section V-B.

**This is not the deployed source.** It is an independently written model whose
purpose is to let a reader of the paper check two published claims by
simulation: that one cycle of the array performs 9 x 16 x 8 = 1152 int8
multiply-accumulates with the reduction running across the taps first and the
input channels second, and that two int8 products sharing one activation
operand leave a +1 artifact on the even lane whenever that activation byte is
zero and the odd-lane coefficient is negative. The line buffering, the DDR-side
addressing, the coefficient double buffering and the requantization path of the
deployed engine are not modelled here, and the deployed RTL is not released.

## Files

| file | what it is |
|---|---|
| `conv3x3_engine.v` | the engine: the MAC array followed by the output stage |
| `conv3x3_mac_array.v` | the array: coefficient store, window formation, 1152 multipliers, tap-first reduction, lane binary tree |
| `conv3x3_mac_pair.v` | one packed multiplier pair, including the artifact |
| `conv3x3_requant.v` | the output stage: group accumulation, `acc * M + bias`, ReLU, output zero-point, 16-bit round-half-up shift, saturation to a byte |
| `tb_mac_selfcheck.v` | self-checking testbench for the array |
| `tb_engine_selfcheck.v` | self-checking testbench for the whole engine |
| `sim/selfcheck_log.txt` | the captured array run |
| `sim/engine_selfcheck_log.txt` | the captured engine run |
| `sim/mac_array.vcd`, `sim/engine.vcd` | the waveforms, openable in any viewer |
| `sim/mac_array_overview.png` | timing panel: coefficient load, then the activation stream |
| `sim/mac_array_artifact.png` | timing panel: the packed-multiplier artifact in isolation |
| `sim/engine_overview.png` | timing panel: two groups accumulated, then requantized |

## How it is checked

`tb_mac_selfcheck.v` computes the expected partial sums independently, with a
plain nested loop over the nine taps and the sixteen input channels, and latches
the array's outputs the way a consumer would, on `psum_we`, so a misplaced valid
cannot pass unnoticed. Over 560 activation beats it compares all 558 complete
windows position by position:

```
  multiply-accumulates per cycle : 1152
  complete windows expected      : 558
  psum_we strobes observed       : 558
  packed-DSP artifact terms      : 58659
  MISMATCHES                     : 0
  RESULT: PASS
```

The stimulus mixes random coefficients and activations with about a third of the
activation bytes forced to zero, a phase in which every activation byte is zero,
and a phase pinning the coefficient extremes -128 and +127 against activations 0
and 255.

`sim/mac_array_artifact.png` shows the artifact directly. While every activation
byte is zero the true product sum is zero, so each even-lane output settles at
the number of negative odd-lane coefficients paired with it (91 and 53 in that
run) while the odd lanes read exactly 0.

The authors additionally co-simulated this model against the deployed engine
with shared stimulus, including directed corner cases, and found the two
datapaths to agree on every compared sample. That testbench instantiates the
deployed source, so it is not part of this release and that particular check
cannot be repeated from the material here; the self-checks above can.

### The whole engine

`tb_engine_selfcheck.v` drives a two-group convolution, 32 input channels over the
same output positions, and compares every requantized byte against a golden that
performs the convolution and the requantization independently:

```
  input channels                 : 32 (2 groups)
  output positions expected      : 78
  dout_we strobes observed       : 78
  packed-DSP artifact terms      : 13219
  saturated high / low           : 290 / 0
  MISMATCHES                     : 0
  RESULT: PASS
```

The authors additionally co-simulated the output stage against the deployed one
over six passes that swept the multiplier from 1 to 65535 and the output
zero-point over 0, 64, 128 and 255, including a two-group accumulation, and
found the two to agree on all 384 output words. That testbench instantiates the
deployed source, so it is not part of this release.

## Interface contract

The model differs from the deployed engine in ways that are deliberate and that
a reader should know about:

* A coefficient burst is exactly eight contiguous beats, one per output channel;
  a gap in `coef_we` restarts the burst at output channel 0.
* Coefficients may not be reloaded while an activation stream is in flight. The
  model holds one coefficient set; the deployed engine carries a shadow buffer
  for this case, which is not modelled.
* The three rows of the window arrive together, one NHWC-16 word each, and the
  three columns are formed from three consecutive beats.
* The pipeline latency of this model is not the pipeline latency of the deployed
  engine. `psum_we` marks the sample it accompanies in this model; that is what
  the self-check verifies.
* In the output stage ReLU is applied in the scaled domain, before the output
  zero-point is added, so a clamp at zero is what implements the activation
  function; the article states the convention but not this ordering.
* The output stage saturates into [0, 255] after the round-half-up shift, and
  the round is taken on the 16-bit binary point of the scaled accumulator.
