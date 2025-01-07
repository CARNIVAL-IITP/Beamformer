import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from .FFT import ConviSTFT, ConvSTFT
from .fspen import FullSubPathExtension


class Total_model(nn.Module):
    def __init__(self, args):
        super(Total_model, self).__init__()

        self.args=args
        self.STFT=ConvSTFT(**args['FFT'])
        self.iSTFT=ConviSTFT(**args['FFT'])


        self.groups=args["dual_path_extension"]["parameters"]["groups"]
        self.inter_hidden_size = self.args["dual_path_extension"]["parameters"]["inter_hidden_size"]
        self.num_modules = self.args["dual_path_extension"]["num_modules"]
        self.num_bands = sum(self.args["bands_num_in_groups"])

        self.model=FullSubPathExtension(args)

    def forward(self, input_wav):
        # print(input_wav.device)
        original_len=input_wav.shape[-1]
        batch_size=input_wav.shape[0]

        input_stft_r, input_stft_i=self.STFT(input_wav, cplx=True)

      

        complex_spectrum=torch.view_as_complex(torch.stack([input_stft_r, input_stft_i], dim=-1))

        # complex_spectrum=complex_spectrum[:,0:1]
    
        
        amplitude_spectrum = torch.abs(complex_spectrum)

        complex_spectrum = torch.view_as_real(complex_spectrum)  # (B, C, F, T, 2)
        
        complex_spectrum = torch.permute(complex_spectrum, dims=(0,1, 3, 4, 2))
        _, mic_channel, frames, channels, frequency = complex_spectrum.shape # (B, C, T, 2, F)
 
        complex_spectrum = torch.reshape(complex_spectrum, shape=(batch_size, mic_channel, frames, channels, frequency))
        amplitude_spectrum = torch.permute(amplitude_spectrum, dims=(0, 1, 3, 2))
        amplitude_spectrum = torch.reshape(amplitude_spectrum, shape=(batch_size, mic_channel, frames, 1, frequency))

        in_hidden_state = [[torch.zeros(1, batch_size * self.num_bands, self.inter_hidden_size//self.groups).to(input_wav.device) for _ in range(self.groups)]
                       for _ in range(self.num_modules)]
        # in_hidden_state=in_hidden_state.to(input_wav.device)
        out, _=self.model(complex_spectrum, amplitude_spectrum, in_hidden_state)

        out=out.permute(0,3,1,2)
        out_real=out[...,0]
        out_imag=out[...,1]

        

        
        out_wav=self.iSTFT(out_real, out_imag, cplx=True)
        out_wav=out_wav[..., :original_len]

        
      
        return out_wav


