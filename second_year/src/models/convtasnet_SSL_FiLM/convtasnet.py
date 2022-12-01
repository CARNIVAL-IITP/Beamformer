from torch.nn.modules import conv
from torch import nn
import torch
from util import *
import matplotlib.pyplot as plt
import numpy as np
from .Causal_CRN_SPL_target import CRN_main
from .convtasnet_module import conv_tasnet



class main_model(nn.Module):
    def __init__(self, config):
        super(main_model, self).__init__()
        self.config=config
        
        self.eps=np.finfo(np.float32).eps
        ### CRN
        self.CRN=CRN_main.get_model(self.config['CRN'])
        

        ### convtasnet
        self.convtasnet=conv_tasnet.TasNet(**self.config['TasNet'])

       





    def irtf_featue(self, x, target):
        r, i, target =self.stft_model(x, target, cplx=True)
       
        
       
        comp = torch.complex(r, i)
        
        comp_ref = comp[..., [self.ref_ch], :, :]
        comp_ref = torch.complex(
        comp_ref.real.clamp(self.eps), comp_ref.imag.clamp(self.eps)
        )

        comp=torch.cat(
        (comp[..., self.ref_ch-1:self.ref_ch, :, :], comp[..., self.ref_ch+1:, :, :]),
        dim=-3) / comp_ref
        x=torch.cat((comp.real, comp.imag), dim=1)

        return x, target
    

    
    def forward(self, x, ):

        ssl_condition=self.CRN(x)
        x=self.convtasnet(x, ssl_condition)
       
        
        return x



