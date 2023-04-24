import torch.optim as optim

from voxelnet.utils.registry import SCHEDULERS

@SCHEDULERS.register_module()
class LambdaStepLR(optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


@SCHEDULERS.register_module()
class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
        super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)

