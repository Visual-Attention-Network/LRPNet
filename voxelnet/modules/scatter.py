import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from voxelnet.modules.load import voxel_module

class ScatterFunction(Function):

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                index: torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        index = index.contiguous()

        output,grad_indices = voxel_module.scatter_max_cuda(input,index)

        ctx.for_backwards = (grad_indices,)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, = ctx.for_backwards
        grad_input = voxel_module.scatter_backward_cuda(grad_output,indices)

        return grad_input, None

voxel_scatter_max = ScatterFunction.apply


