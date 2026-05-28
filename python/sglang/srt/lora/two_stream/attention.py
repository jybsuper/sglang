"""Side-stream LoRA forward implementations for attention projections (O7, O8).

These are standalone functions (not methods) so the corresponding overrides
in ``lora/layers.py`` are tiny delegates — the bulk of the two-stream logic
lives here.

The overlap pattern is the same for both projections:

    side stream:  lora_a_shrink(input)           # cheap GEMM on small rank
    main stream:  base_layer.quant_method.apply(input)   # full FP8 GEMM
    -- rejoin --
    main stream:  lora_b_expand(shrink_intermediate, base_output)  # atomic-add

The shrink and the base GEMM read the same ``input`` tensor (no write
conflict) so they can race safely. The expand requires both outputs, so it
serializes after the join on the main stream.
"""
from typing import Optional

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.lora.two_stream import get_lora_side_stream, is_two_stream_active


def qkv_proj_lora_forward(layer, input_: torch.Tensor):
    """O7: QKVParallelLinearWithLoRA forward override.

    Side-stream LoRA-A shrink (``sgemm_lora_a_fwd`` with ``stack_num=3``)
    concurrent with the base qkv_proj GEMM; rejoin before
    ``qkv_lora_b_fwd`` atomic-adds the delta to base_output.

    Falls back to the base ColumnParallel forward when two-stream is
    inactive or LoRA isn't set, so it can stand in for the inherited
    ``forward`` method on every call.
    """
    # Late import to avoid a layers.py <-> two_stream import cycle.
    from sglang.srt.lora.layers import ColumnParallelLinearWithLoRA
    from sglang.srt.lora.triton_ops import qkv_lora_b_fwd, sgemm_lora_a_fwd

    if not layer.set_lora or not is_two_stream_active(input_):
        return ColumnParallelLinearWithLoRA.forward(layer, input_)

    bias = layer.base_layer.bias if not layer.base_layer.skip_bias_add else None
    side_stream = get_lora_side_stream()
    # sgemm_info is host-side (LoRABatchInfo); compute once, share both calls.
    sgemm_info = layer.lora_backend._sgemm_info()

    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        shrink_intermediate = sgemm_lora_a_fwd(
            input_, layer.A_buffer_qkv, sgemm_info, stack_num=3
        )

    # Base qkv_proj GEMM on main, concurrent with the side-stream shrink.
    output_parallel = layer.base_layer.quant_method.apply(
        layer.base_layer, input_, bias
    )

    # Join: expand reads both the side-produced shrink_intermediate and
    # base_output, so wait for the side stream before launching it.
    torch.cuda.current_stream().wait_stream(side_stream)
    output_parallel = qkv_lora_b_fwd(
        shrink_intermediate,
        layer.B_buffer_qkv,
        sgemm_info,
        layer.output_offset,
        layer.max_qkv_out_dim,
        output_parallel,
        n_slices=3,
    )

    if layer.base_layer.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = layer.base_layer.bias if layer.base_layer.skip_bias_add else None
    return output, output_bias


def row_parallel_lora_forward(
    layer, input_: torch.Tensor, skip_all_reduce=False, forward_batch=None
):
    """O8: RowParallelLinearWithLoRA forward override.

    Same overlap pattern as O7, with row-parallel specifics:

      - input is split along the last dim per TP rank (unless already parallel)
      - bias is rank-0 only
      - after the join, optional all-reduce on the base output (and on the
        side-produced ``lora_a_output`` if reducing) then LoRA-B expand
        atomic-adds to base; final all-reduce when needed
    """
    if layer.base_layer.input_is_parallel:
        input_parallel = input_
    else:
        tp_rank = get_tensor_model_parallel_rank()
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=layer.base_layer.tp_size
        )
        input_parallel = splitted_input[tp_rank].contiguous()

    bias_ = (
        None
        if (layer.base_layer.tp_rank > 0 or layer.base_layer.skip_bias_add)
        else layer.base_layer.bias
    )

    # Fork the side-stream LoRA-A shrink (decode-only).
    two_stream = layer.set_lora and is_two_stream_active(input_parallel)
    lora_a_output: Optional[torch.Tensor] = None
    side_stream = None
    if two_stream:
        side_stream = get_lora_side_stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            lora_a_output = layer.lora_backend.run_lora_a_sgemm(
                input_parallel, layer.A_buffer
            )

    # Base row-parallel GEMM on main, concurrent with the side-stream shrink.
    output_parallel = layer.base_layer.quant_method.apply(
        layer.base_layer, input_parallel, bias=bias_
    )

    if two_stream:
        torch.cuda.current_stream().wait_stream(side_stream)

    should_reduce = (
        layer.base_layer.reduce_results
        and layer.base_layer.tp_size > 1
        and not skip_all_reduce
    )

    if layer.set_lora and should_reduce:
        if lora_a_output is None:
            lora_a_output = layer.lora_backend.run_lora_a_sgemm(
                input_parallel, layer.A_buffer
            )
        output_ = tensor_model_parallel_all_reduce(output_parallel)
        lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)
        output_ = layer.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=layer.B_buffer,
            output_offset=layer.output_offset,
            output_offset_cpu=layer.output_offset_cpu,
            base_output=output_,
        )
    else:
        if layer.set_lora:
            if lora_a_output is not None:
                # Two-stream branch already ran the shrink on the side stream;
                # finish the LoRA with just the expand against output_parallel.
                output_parallel = layer.lora_backend.run_lora_b_sgemm(
                    x=lora_a_output,
                    weights=layer.B_buffer,
                    output_offset=layer.output_offset,
                    output_offset_cpu=layer.output_offset_cpu,
                    base_output=output_parallel,
                )
            else:
                output_parallel = layer.apply_lora(output_parallel, input_parallel)
        if should_reduce:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

    output_bias = layer.base_layer.bias if layer.base_layer.skip_bias_add else None
    return output_, output_bias
