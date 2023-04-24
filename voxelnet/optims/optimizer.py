
from voxelnet import optims

import torch.optim as optim

from voxelnet.utils.registry import OPTIMS

@OPTIMS.register_module()
class SGD(optim.SGD):
    def __init__(self, params, lr=1e-1, momentum=0.9, dampening=0.1,
                 weight_decay=1e-4, nesterov=False):
        super().__init__(params,lr,momentum,dampening,weight_decay,nesterov)