import torch
import numpy as np
from torch.nn.modules.loss import _Loss

from asteroid.losses import singlesrc_neg_sisdr
from asteroid.losses.sdr import SingleSrcNegSDR


class sync_SI_SDR(_Loss):
    def __init__(self, reduction="none"):
        super(sync_SI_SDR, self).__init__()
        self.reduction=reduction
        self.EPS=1e-8
        self.func_for_loss=SingleSrcNegSDR(reduction=reduction, zero_mean=True, take_log=True, sdr_type='sisdr')

        


    def forward(self, output, target):


        loss=self.func_for_loss(output, target)

        return loss
        
