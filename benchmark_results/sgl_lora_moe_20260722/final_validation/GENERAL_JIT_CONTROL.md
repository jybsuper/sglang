# GB300 general FP8 JIT/AOT control

This diagnostic is outside the LoRA acceptance matrix, but it is retained because
the terminal run intentionally exercised a broader changed test file.

The full GB300 combined shard reported 135 passes and 66 failures. Every failure was
in the pre-existing unmasked `test_v2_jit_matches_aot` bit-exact comparison; the
branch-added masked-layout tests and all LoRA quant-provider plan tests passed in the
focused `g2.log` (41 passed). The same broader H200 shard passed 200 cases with one
expected Blackwell-only skip.

To separate an existing architecture issue from the LoRA change, one representative
failing GB300 node was rerun with the official-main AOT source at
`4eaa5ca6510622cb0006bcfee5947b17859ac8c7`:

```text
test_v2_jit_matches_aot[dtype6-1-2048-True-False]
```

It failed with the same `fp8 codes differ` assertion. The exact AOT source hashes
used on the device were:

```text
40488688613fea24e5dae2d1cd0f61371ce57a937bcac049fa1a4b3c8da660cd  official main
a1c56b78301d7d0ad749fe618d1ecebabd8aab2414f210b62409bed10ddc0e81  LoRA branch
```

The LoRA diff changes only masked-layout dispatch and subwarp selection; the failing
node uses the unchanged unmasked `NaiveScheduler`. This makes the control sufficient
to classify the result as an upstream GB300 JIT/AOT numerical-equivalence issue,
while keeping it visible for later general quant-kernel work.

Raw logs:

- `gb300/optional_general_jit_aot_gb300.log`: broad branch run.
- `gb300/upstream_general_jit_control.log`: representative official-main control.
- `h200/optional_general_jit_aot_h200.log`: broader H200 passing run.
