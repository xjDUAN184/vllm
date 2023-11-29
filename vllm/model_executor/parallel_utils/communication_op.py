import torch
import torch.distributed._functional_collectives as distfunc

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)


def tensor_model_parallel_all_reduce(
    input_: torch.Tensor,
    reduce_op: str = "sum",
) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: Unlike `torch.distributed.all_reduce`, this operation is not applied
    in-place and can be captured by CUDA graph.
    """
    tp_world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if tp_world_size == 1:
        return input_
    return distfunc.all_reduce(input_, reduce_op, group=get_tensor_model_parallel_group())


def tensor_model_parallel_all_gather(input_, dim=-1):
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor
