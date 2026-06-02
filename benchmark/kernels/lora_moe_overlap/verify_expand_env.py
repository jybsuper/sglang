"""E2E correctness check for the SGLANG_LORA_EXPAND_BLOCK_M wiring.

Runs the real `merged_experts_fused_moe_lora_add` (gate_up stage, which takes the
direct-expand path through the env code) with the knob OFF and ON, and asserts:
  - env-ON output == env-OFF output  (the BLOCK_SIZE_M / BLOCK_SIZE_N override is a
    pure tiling change -> must be numerically invariant), and
  - both match the torch reference (the path itself is correct).

Usage: python verify_expand_env.py [--model qwen35] [--ts 64,512] [--block-m 16]
"""
import argparse

import torch

from sglang.srt.environ import envs
from testbed import MODELS, make_inputs, run_e2e_gate_up, torch_ref


def _err(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item(), b.abs().max().item() + 1e-6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="qwen35")
    p.add_argument("--ts", default="64,512")
    p.add_argument("--block-m", type=int, default=16)
    p.add_argument("--max-loras", type=int, default=1)
    args = p.parse_args()
    cfg = MODELS[args.model]
    Ts = [int(x) for x in args.ts.split(",")]
    print(f"== {args.model} e2e env-wiring verify (rank={cfg.lora_rank}, E={cfg.per_gpu_experts}, "
          f"gate_up N={cfg.n_out}, block_m={args.block_m}) ==")

    all_ok = True
    for T in Ts:
        inp = make_inputs(cfg, T, args.max_loras, torch.bfloat16)
        ref = torch_ref(inp, sum_over_topk=False, mul_routed_weight=False)

        envs.SGLANG_LORA_EXPAND_BLOCK_M.clear()
        out_off = run_e2e_gate_up(inp)

        envs.SGLANG_LORA_EXPAND_BLOCK_M.set(args.block_m)
        out_on = run_e2e_gate_up(inp)
        envs.SGLANG_LORA_EXPAND_BLOCK_M.clear()

        d_onoff, _ = _err(out_on, out_off)
        e_off, den = _err(out_off, ref)
        e_on, _ = _err(out_on, ref)
        ok = (d_onoff < 1e-3) and torch.allclose(out_on.float(), ref.float(), atol=1e-2, rtol=1e-2)
        all_ok &= ok
        print(f"  T={T:<5d}: ON-vs-OFF max_abs={d_onoff:.2e} | OFF-vs-torch rel={e_off/den:.2e} "
              f"| ON-vs-torch rel={e_on/den:.2e} -> {'OK' if ok else 'FAIL'}")
    print("RESULT:", "ALL OK" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
