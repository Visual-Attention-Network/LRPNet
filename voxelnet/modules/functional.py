from typing import Callable

import torch

from torchsparse import SparseTensor

def unique(x,sorted=True, return_inverse=False, return_counts=False,return_index=False, dim=None):
    if return_index:
        r_in = True 
    else:
        r_in = return_inverse
    rets = torch.unique(x, sorted=sorted, return_inverse=r_in, return_counts=return_counts, dim=dim)
    rets = list(rets)
    if return_index:
        inverse = rets[1]
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(rets[0].size(0)).scatter_(0, inverse, perm)
        rets.insert(1,perm)
        if not return_inverse:
            del rets[2]
    return rets 


def fapply(input: SparseTensor, fn: Callable[..., torch.Tensor], *args,
           **kwargs) -> SparseTensor:
    feats = fn(input.feats, *args, **kwargs)
    output = SparseTensor(coords=input.coords, feats=feats, stride=input.stride)
    output.cmaps = input.cmaps
    output.kmaps = input.kmaps
    return output