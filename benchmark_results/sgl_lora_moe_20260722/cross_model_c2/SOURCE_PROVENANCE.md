# Cross-model C2 source provenance

The H200 and GB300 matrices used the same remote working-tree snapshot on base
commit `4ffaee0b413fae1098054a0c98eeae02ab537c02`. The benchmark was intentionally
run from an uncommitted experimental source tree, so the content hashes below,
not the base commit alone, bind the result artifacts to their implementation.

```text
c8c39418be27c49e4e6e5b3dd7ffb2dbba498c11b01495949ea94f11468ae9dc  benchmark/kernels/lora_moe/bench_c2_cross_model_contracts.py
b278ba4fdb25cbd30007c0802f02451bfa524a1138d9b5320b65d1de33617d67  benchmark/kernels/lora_moe/c2_semantic_contracts.py
cfe921e2176857dcee1af8bd33aec08ad25a2d711dab155e470edca21b32799b  benchmark/kernels/lora_moe/run_c2_cross_model_guardrail_shard.sh
472fa9ce27655e318b9061e715ed66531837bccc7ee11616c9a29ec0ed7f5501  benchmark/kernels/lora_moe/matrix.py
9f360e2f1806c73c60935143e9a0ad7ac6adc756712ca7aaf46bc99c0d17772f  benchmark/kernels/lora_moe/profiling.py
daeb01fb00f1a376bc3335e6d1f464967548c5adf0e189dd77314f892b1972b9  python/sglang/srt/lora/sgl_lora/triton_ops/fused_c2.py
300fcd9bffe1d449069fb7ec69965036405ca9f9004aa8077764276b2b21c83b  python/sglang/srt/lora/sgl_lora/triton_ops/fused_c2_relu2.py
```

Both nodes independently reported this identical hash set. After the matrix
started, the local files received only audit metadata/comment/docstring edits:
the benchmark now emits an explicit launch-topology label, and comments state
the already-tested local-EP and base-only contracts. Kernel behavior measured
by the matrix did not change. The registered GPU suites and H200 full-pipeline
regression were rerun after the final behavioral fixes.
