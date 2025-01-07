import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch as th
import torch
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt

EPSILON = th.finfo(th.float32).eps
MATH_PI = math.pi
def init_kernels(win_len,
                 win_inc,
                 fft_len,
                 win_type=None,
                 invers=False,
                 sqrt_window=False):
    if win_type == 'None' or win_type is None:
        # N 
        window = np.ones(win_len)
    else:
        # N
        window = get_window(win_type, win_len, fftbins=True)#**0.5

 

    if sqrt_window:
        window = np.sqrt(window)

    N = fft_len
    # N x F
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    # N x F
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    # 2F x N
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    # 2F x N * N => 2F x N
    kernel = kernel*window
    # 2F x 1 x N
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, 
                 win_len,
                 win_inc,
                 fft_len=None,
                 vad_threshold=2/3,
                 win_type='hamming',
                 sqrt_window=False
                #  fix=True
                 ):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        # 2F x 1 x N
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type, sqrt_window=sqrt_window)
        get_target_kernel=torch.ones_like(kernel).mean(dim=0, keepdim=True)/kernel.shape[-1]
        # print(kernel.shape, get_target_kernel.shape)
        # exit()
        self.register_buffer('weight', kernel)
        self.register_buffer('target_kernel', get_target_kernel)

        
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs, cplx=False):
        
        
        if inputs.dim() == 2:
            # N x 1 x L
            inputs = torch.unsqueeze(inputs, 1)
            inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
            # N x 2F x T
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)
            # N x F x T
            r, i = th.chunk(outputs, 2, dim=1)
        else:
            
            N, C, L = inputs.shape
            inputs = inputs.view(N * C, 1, L)
            # NC x 1 x L
            inputs = F.pad(inputs, [self.win_len-self.stride, self.win_len-self.stride])
            

          
            
            # NC x 2F x T
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)

            # N x C x 2F x T
            outputs = outputs.view(N, C, -1, outputs.shape[-1])
            
            # N x C x F x T
            r, i = th.chunk(outputs, 2, dim=2)
            

        if cplx:
            return r, i
        else:
            # mags = th.clamp(r**2 + i**2, EPSILON)**0.5
            # phase = th.atan2(i+EPSILON, r+EPSILON)
            # return mags, phase, r, i
        
            mags=torch.pow(r**2+i**2, 0.5)
            mags=th.clamp(mags, 1e-12)
            
            phase=torch.atan2(i, r)
            return mags, phase, r, i
    

    def get_target_frame(self, inputs, threshold=0.5):
        inputs = torch.unsqueeze(inputs, 1)

        inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
            # N x 2F x T
        outputs = F.conv1d(inputs, self.target_kernel, stride=self.stride).squeeze(1)
        # plt.plot(outputs[0,0].detach().cpu().numpy())
        # plt.plot(outputs[1,0].detach().cpu().numpy())
        # plt.plot(outputs[2,0].detach().cpu().numpy())
        # plt.plot(outputs[3,0].detach().cpu().numpy())
        # plt.savefig('../results/pngs/frames.png')
        # print(outputs.shape)
        # exit()
        outputs=torch.ge(outputs, threshold).float()

        # outputs=outputs.squeeze().detach().cpu().numpy()
        return outputs
class ConviSTFT(nn.Module):

    def __init__(self, 
                 win_len, 
                 win_inc, 
                 fft_len=None, 
                 win_type='hamming', 
                 sqrt_window=False
                #  fix=True
                 ):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        # kernel: 2F x 1 x N
        # window: 1 x N x 1
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True, sqrt_window=sqrt_window)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase, cplx=False):
        """
        inputs : [B, N//2+1, T] (mags, real)
        phase: [B, N//2+1, T] (phase, imag)
        """ 

        if cplx:
            # N x 2F x T
            cspec = torch.cat([inputs, phase], dim=1)
        else:
            # N x F x T
            real = inputs*torch.cos(phase)
            imag = inputs*torch.sin(phase)
            # N x 2F x T
            cspec = torch.cat([real, imag], dim=1)
        # N x 1 x L
        outputs = F.conv_transpose1d(cspec, self.weight, stride=self.stride)
        
        # this is from torch-stft: https://github.com/pseeth/torch-stft
        # 1 x N x T
        t = self.window.repeat(1,1,inputs.size(-1))**2
        # 1 x 1 x L
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        
        outputs = outputs/(coff+1e-12)
        
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        # N x 1 x L
        
        outputs = outputs[...,self.win_len-self.stride:]
        # N x L
        outputs = outputs.squeeze(1)
        return outputs