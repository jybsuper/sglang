# Source provenance

The GPU runs used isolated repository trees rather than the shared development
checkout:

- H200: `/workspace/sglang_quant_provider_lane`, GPU 7, isolated base commit
  `4ffaee0b413fae1098054a0c98eeae02ab537c02`.
- GB300: `/mirror/sglang_quant_provider_lane`, GPU 3.

The modified worktrees were not represented by a single commit while evidence
was collected, so content hashes are the authoritative provenance:

| Content used by GPU runs | SHA-256 |
| --- | --- |
| `base_gemm.py` provider implementation | `cd3bb9ed5c080b5d7c9125dde5fb2c702d1e2e77c48a77e27708dab2697bc6b8` |
| `quant_info.py` provider payloads | `81eef385527a057c4112048d278d459a5b62cc22e95e661e40a709bfd7d1dc0a` |
| eager H200 harness | `8663bc6b583037b0b839ded0c42cf7640a17be122aa87533371ab6886e5d648f` |
| graph H200 and eager/graph GB300 harness | `8895aa41372d3e7aed243f05141e84727546e9ef6aa52907362a0c6fa2378605` |

The harness difference was the addition of direct CUDA graph capture/replay;
the provider semantics and eager invocation were unchanged. After evidence was
collected, Black formatting and a workspace-doc clarification changed the
provider file hash without changing executable semantics. The reviewed local
sources at packaging time are:

| Reviewed source | SHA-256 |
| --- | --- |
| `base_gemm.py` | `fd67737ac84c14b4476b091a2cf7fb4f1b017a106b49a3c315f89f0bf8d23faa` |
| `quant_info.py` | `81eef385527a057c4112048d278d459a5b62cc22e95e661e40a709bfd7d1dc0a` |
| `moe_lora_runner.py` | `cdea44cf11810003e6a8b1e8814161246d758de8146b7c4cd6b934df39e55143` |
| `lora_layer.py` | `d570d4682fa8b96be03c9cbeac5ab96f9c2fc8460d332399ef77993906a9f52d` |
| `base_backend.py` | `ed52cf6f2cb58829a474cd0d9424b0f8e6f69cfa15e16737971720fb527fa4ac` |
| `bench_quant_providers.py` | `cb5ac30353002418ee6002fc09fedd795251486ae6ca236fc26fff6c162edb35` |
| `test_sgl_lora_quant_providers.py` | `d1549eb8c179c31b7c9555b051ac56606b419d7e9cced415acf70191459877c9` |
| `test_sgl_lora_runner.py` | `a27ba3353750a461c8817cc979601dceb3625561839b183306ee6d28a431c756` |

The direct provider harness does not exercise attach-time provider discovery or
the complete virtual-expert runner. Those boundaries are covered by the focused
tests recorded in `h200/test_summary.json`.

Hardware/software recorded by the raw JSON:

- H200: compute capability 9.0, PyTorch `2.11.0+cu129`, Python `3.12.3`.
- GB300: compute capability 10.3, PyTorch `2.11.0+cu130`, Python `3.12.3`.

No file under `python/sglang/srt/lora/sgl_lora` imports
`sglang.jit_kernel.trtllm_lora_temp`.
