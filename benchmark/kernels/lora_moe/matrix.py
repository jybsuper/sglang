"""Named, curated MoE-LoRA benchmark matrices."""

from __future__ import annotations

from .cases import AdapterBatch, Device, ModelShape, MoeLoraBenchCase

MODEL_PRESETS: dict[str, ModelShape] = {
    "qwen3.5-35b-a3b": ModelShape(
        key="qwen3.5-35b-a3b",
        h_model=2048,
        h_moe=2048,
        intermediate_size=512,
        num_experts=256,
        top_k=8,
        num_slices=2,
        activation="swiglu",
        moe_layers=40,
    ),
    "qwen3.5-397b-a17b": ModelShape(
        key="qwen3.5-397b-a17b",
        h_model=4096,
        h_moe=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        num_slices=2,
        activation="swiglu",
        moe_layers=60,
    ),
    "kimi-k2.5": ModelShape(
        key="kimi-k2.5",
        h_model=7168,
        h_moe=7168,
        intermediate_size=2048,
        num_experts=384,
        top_k=8,
        num_slices=2,
        activation="swiglu",
        moe_layers=60,
    ),
    "glm-5.2": ModelShape(
        key="glm-5.2",
        h_model=6144,
        h_moe=6144,
        intermediate_size=2048,
        num_experts=256,
        top_k=8,
        num_slices=2,
        activation="swiglu",
        moe_layers=75,
    ),
    "nemotron-3-super": ModelShape(
        key="nemotron-3-super",
        h_model=4096,
        h_moe=1024,
        intermediate_size=2688,
        num_experts=512,
        top_k=22,
        num_slices=1,
        activation="relu2",
        moe_layers=40,
    ),
    "nemotron-3-nano": ModelShape(
        key="nemotron-3-nano",
        h_model=2688,
        h_moe=2688,
        intermediate_size=1856,
        num_experts=128,
        top_k=6,
        num_slices=1,
        activation="relu2",
        moe_layers=23,
    ),
}


# (id, T, L_active, B_base, L_capacity, R, phase, cache state)
P0_CELLS: tuple[tuple[str, int, int, int, int, int, str, str], ...] = (
    ("tiny", 1, 1, 0, 1, 32, "decode", "cold"),
    ("base", 32, 0, 1, 8, 64, "decode", "cold"),
    ("cap1", 32, 1, 0, 1, 64, "decode", "cold"),
    ("sparse", 32, 1, 0, 8, 64, "decode", "cold"),
    ("mixed", 32, 1, 1, 8, 64, "decode", "cold"),
    ("odd-full", 32, 4, 1, 5, 64, "decode", "cold"),
    ("default-mixed-full", 32, 7, 1, 8, 128, "decode", "cold"),
    ("default-lora-full", 32, 8, 0, 8, 128, "decode", "cold"),
    ("decode-large", 256, 1, 0, 1, 32, "decode", "cold"),
    ("prefill-small", 128, 3, 0, 8, 64, "prefill", "cold"),
    ("prefill-threshold", 256, 3, 0, 8, 64, "prefill", "cold"),
    ("prefill-threshold-plus", 257, 3, 0, 8, 64, "prefill", "cold"),
    ("prefill", 2048, 3, 0, 8, 64, "prefill", "cold"),
    ("routing-hot", 256, 8, 0, 8, 128, "decode", "hot"),
)


def get_model_shape(model_key: str) -> ModelShape:
    try:
        return MODEL_PRESETS[model_key]
    except KeyError as exc:
        choices = ", ".join(MODEL_PRESETS)
        raise ValueError(
            f"unknown model preset {model_key!r}; choose from {choices}"
        ) from exc


def p0_cases(
    device: Device,
    *,
    model_key: str = "qwen3.5-35b-a3b",
) -> tuple[MoeLoraBenchCase, ...]:
    model = get_model_shape(model_key)
    cases = []
    for cell_id, tokens, active, base, capacity, rank, phase, cache in P0_CELLS:
        cases.append(
            MoeLoraBenchCase(
                case_id=f"p0-{model.key}-{cell_id}-{device}",
                model=model,
                adapters=AdapterBatch(
                    l_active=active,
                    b_base=base,
                    l_capacity=capacity,
                    rank=rank,
                    max_rank=rank,
                    physical_rank=rank,
                ),
                t_local=tokens,
                phase=phase,
                device=device,
                provider="deepgemm_bf16",
                scope="M0",
                stage="M0",
                pipeline="N0" if active == 0 else "C0",
                graph_mode="eager",
                routing="deterministic_lattice",
                cache_state=cache,
            )
        )
    return tuple(cases)


def model_shape_cases(device: Device) -> tuple[MoeLoraBenchCase, ...]:
    """One cheap resolved local-shape case for each supported model preset."""

    cases = []
    for model in MODEL_PRESETS.values():
        cases.append(
            MoeLoraBenchCase(
                case_id=f"shape-{model.key}-{device}",
                model=model,
                adapters=AdapterBatch(
                    l_active=1,
                    b_base=0,
                    l_capacity=8,
                    rank=32,
                    max_rank=32,
                    physical_rank=32,
                ),
                t_local=32,
                phase="decode",
                device=device,
                provider="deepgemm_bf16",
                scope="M0",
                stage="M0",
                pipeline="C0",
                graph_mode="eager",
                routing="deterministic_lattice",
                cache_state="cold",
            )
        )
    return tuple(cases)
