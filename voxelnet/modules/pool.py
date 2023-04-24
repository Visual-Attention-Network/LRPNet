from typing import Optional, Tuple, Union

import torch
from torch import nn 
import numpy as np 

from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple

from voxelnet.modules.scatter import voxel_scatter_max


def get_kernel_offsets(size: Union[int, Tuple[int, ...]],
                       stride: Union[int, Tuple[int, ...]] = 1,
                       dilation: Union[int, Tuple[int, ...]] = 1,
                       device: str = 'cpu') -> torch.Tensor:
    size = make_ntuple(size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    offsets = [(np.arange(-size[k] // 2 + 1, size[k] // 2 + 1) * stride[k]
                * dilation[k]) for k in range(3)]

    # This condition check is only to make sure that our weight layout is
    # compatible with `MinkowskiEngine`.
    if np.prod(size) % 2 == 1:
        offsets = [[x, y, z] for z in offsets[2] for y in offsets[1]
                   for x in offsets[0]]
    else:
        offsets = [[x, y, z] for x in offsets[0] for y in offsets[1]
                   for z in offsets[2]]

    offsets = torch.tensor(offsets, dtype=torch.int, device=device)
    return offsets


def pool(input: SparseTensor,
           kernel_size: Union[int, Tuple[int, ...]],
           stride: Union[int, Tuple[int, ...]] = 1,
           dilation: Union[int, Tuple[int, ...]] = 1) -> SparseTensor:
    feats, coords = input.feats, input.coords

    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    if (kernel_size == (1, 1, 1) and stride == (1, 1, 1)
            and dilation == (1, 1, 1)):
        output = SparseTensor(coords=coords, feats=feats, stride=input.stride)
    else:
        kmap = input.kmaps.get((input.stride, kernel_size, stride, dilation))
        if kmap is None:
            offsets = get_kernel_offsets(kernel_size,
                                         dilation=dilation,
                                         stride=input.stride,
                                         device=feats.device)

            references = F.sphash(coords)
            if any(s > 1 for s in stride):
                coords = F.spdownsample(coords, stride, kernel_size,
                                        input.stride)
            queries = F.sphash(coords, offsets)
            results = F.sphashquery(queries, references) #[K,N]

            nbsizes = torch.sum(results != -1, dim=1)
            nbmaps = torch.nonzero(results != -1)
            nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1)
                                            + nbmaps[:, 1]]
                

            kmap = [nbmaps, nbsizes, (feats.shape[0], coords.shape[0])]
            input.kmaps[(input.stride, kernel_size, stride, dilation)] = kmap
        
        nbmaps = kmap[0]
        feats = voxel_scatter_max(feats,nbmaps)
        output = SparseTensor(
            coords=coords,
            feats=feats,
            stride=tuple(input.stride[k] * stride[k] for k in range(3)))
    output.cmaps = input.cmaps
    output.cmaps.setdefault(output.stride, output.coords)
    output.kmaps = input.kmaps
    return output
    
class Pool(nn.Module):

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
       
    def forward(self, input: SparseTensor) -> SparseTensor:
        return pool(input,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation)
