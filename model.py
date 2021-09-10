import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch as th
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.signal import get_window
import numpy as np

EPSILON = th.finfo(th.float32).eps
MATH_PI = math.pi

def init_kernels(win_len,
                 win_inc,
                 fft_len,
                 win_type=None,
                 invers=False):
    if win_type == 'None' or win_type is None:
        # N 
        window = np.ones(win_len)
    else:
        # N
        window = get_window(win_type, win_len, fftbins=True)#**0.5
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
                 win_type='hamming',
                #  fix=True
                 ):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        # 2F x 1 x N
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
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
            mags = th.clamp(r**2 + i**2, EPSILON)**0.5
            phase = th.atan2(i+EPSILON, r+EPSILON)
            return mags, phase

class ConviSTFT(nn.Module):

    def __init__(self, 
                 win_len, 
                 win_inc, 
                 fft_len=None, 
                 win_type='hamming', 
                #  fix=True
                 ):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        # kernel: 2F x 1 x N
        # window: 1 x N x 1
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
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
        outputs = outputs/(coff+1e-8)
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        # N x 1 x L
        outputs = outputs[...,self.win_len-self.stride:-(self.win_len-self.stride)]
        # N x L
        outputs = outputs.squeeze(1)
        return outputs

class CRNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(CRNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2))   # 2-ch input : Real, Imag
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=256)

        # LSTM
        # for 640 samples / frame
        # self.LSTM1 = nn.LSTM(input_size=2304, hidden_size=2304, num_layers=2, batch_first=True)
        # # for 320 samples / frame
        self.LSTM1 = nn.LSTM(input_size=1792, hidden_size=1792, num_layers=3, batch_first=True)

        # grouped LSTM ( K=2 )
        # self.LSTM1_1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        # self.LSTM1_2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        # self.LSTM2_1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        # self.LSTM2_2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)


        # Decoder for real
        self.convT1_real = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1_real = nn.BatchNorm2d(num_features=128)
        self.convT2_real = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2_real = nn.BatchNorm2d(num_features=64)
        self.convT3_real = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3_real = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.convT4_real = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2), output_padding=(0, 1))
        self.bnT4_real = nn.BatchNorm2d(num_features=16)
        self.convT5_real = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5_real = nn.BatchNorm2d(num_features=1)
        # Decoder for imag
        self.convT1_imag = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1_imag = nn.BatchNorm2d(num_features=128)
        self.convT2_imag = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2_imag = nn.BatchNorm2d(num_features=64)
        self.convT3_imag = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3_imag = nn.BatchNorm2d(num_features=32)
        # output_padding为1，不然算出来是79
        self.convT4_imag = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2), output_padding=(0, 1))
        self.bnT4_imag = nn.BatchNorm2d(num_features=16)
        self.convT5_imag = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5_imag = nn.BatchNorm2d(num_features=1)
        
      


    def forward(self,x):
        
        # (B, in_c, T, F)
        
        
        x1=self.bn1(self.conv1(x))
        x1=F.elu(x1)
        

        x2=self.bn2(self.conv2(x1))
        x2=F.elu(x2)
        
        x3=self.bn3(self.conv3(x2))
        x3=F.elu(x3)

        x4=self.bn4(self.conv4(x3))
        x4=F.elu(x4)

        x5=self.bn5(self.conv5(x4))
        x5=F.elu(x5)
        
       
       
        # reshape
        out5 = x5.permute(0, 2, 1, 3) # [B, T, Ch, F]
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)        
        lstm2, (hn, cn) = self.LSTM1(out5)
        # print(lstm2.shape)
        # exit()
        # grouped lstm (K=2)
        # lstm1_front, (hn, cn) = self.LSTM1_1(out5[:, :, :512])
        # lstm1_back, (hn2, cn2) = self.LSTM1_2(out5[:, :, 512:])
        #
        # lstm1_front2 = lstm1_front.unsqueeze(3)
        # lstm1_back2 = lstm1_back.unsqueeze(3)
        #
        # lstm1_reshape = torch.cat((lstm1_front2, lstm1_back2), 3)
        # lstm1_trans = torch.transpose(lstm1_reshape, 2, 3)
        # lstm1_rearrange = lstm1_trans.reshape(lstm1_trans.size()[0], lstm1_trans.size()[1], -1)
        # lstm2_front, (_, _) = self.LSTM2_1(lstm1_rearrange[:, :, 0:512])
        # lstm2_back, (_, _) = self.LSTM2_2(lstm1_rearrange[:, :, 512:])
        # lstm2 = torch.cat((lstm2_front, lstm2_back), 2)

        # reshape
        output = lstm2.reshape(lstm2.size()[0], lstm2.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
        # print(output.shape, x5.shape)
        # exit()      

        # ConvTrans for real
        res_real = torch.cat((output, x5), 1)

        res1_real=self.bnT1_real(self.convT1_real(res_real))
        res1_real=F.elu(res1_real)
        res1_real = torch.cat((res1_real, x4), 1)
        
        res2_real=self.bnT2_real(self.convT2_real(res1_real))
        res2_real=F.elu(res2_real)
        res2_real = torch.cat((res2_real, x3), 1)

        res3_real=self.bnT3_real(self.convT3_real(res2_real))
        res3_real=F.elu(res3_real)
        res3_real = torch.cat((res3_real, x2), 1)

        res4_real=self.bnT4_real(self.convT4_real(res3_real))
        res4_real=F.elu(res4_real)
        res4_real = torch.cat((res4_real, x1), 1)
        
        res5_real=self.bnT5_real(self.convT5_real(res4_real))  # [B, 1, T, 161]
        

        # ConvTrans for imag
        res_imag = torch.cat((output, x5), 1)

        res1_imag=self.bnT1_imag(self.convT1_imag(res_imag))
        res1_imag=F.elu(res1_imag)
        res1_imag = torch.cat((res1_imag, x4), 1)

        res2_imag=self.bnT2_imag(self.convT2_imag(res1_imag))
        res2_imag=F.elu(res2_imag)
        res2_imag = torch.cat((res2_imag, x3), 1)

        res3_imag=self.bnT3_imag(self.convT3_imag(res2_imag))
        res3_imag=F.elu(res3_imag)
        res3_imag = torch.cat((res3_imag, x2), 1)

        res4_imag=self.bnT4_imag(self.convT4_imag(res3_imag))
        res4_imag=F.elu(res4_imag)
        res4_imag = torch.cat((res4_imag, x1), 1)
        
        res5_imag=self.bnT5_imag(self.convT5_imag(res4_imag))  # [B, 1, T, 161]


        # can be changed
        res5_real=F.elu(res5_real)*x[:,0:1,:,:]
        res5_imag=F.elu(res5_imag)*x[:,4:5,:,:]

        # res5_real=F.softplus(res5_real)*x[:,0:1,:,:]
        # res5_imag=F.softplus(res5_imag)*x[:,4:5,:,:]
        
       
        res5 = torch.cat((res5_real, res5_imag), 1)     # [B, 2, T, 161]
        
        
        return res5.squeeze()

class FFT_CRN_IMAG(nn.Module):
    def __init__(self, config):
        super(FFT_CRN_IMAG, self).__init__()

        self.stft_module=ConvSTFT(config['window_size'], config['hop_size'], config['window_size'])
        
        self.CRN=CRNN()

        self.istft_module=ConviSTFT(config['window_size'], config['hop_size'], config['window_size'])
    
    def forward(self, x):
        x=self.stft_module(x, cplx=True)
        
        x=torch.cat((x[0], x[1]), dim=1).permute(0,1,3,2)
        
        output_x=self.CRN(x)
        # print(output_x.dim())
        # exit()
        if output_x.dim()==3:
            output_x=output_x.unsqueeze(0)
        output_x=output_x.permute(0,1,3,2) # input과 곱하기 하기
        # print(x.shape)
        # exit()
        x=self.istft_module(output_x[:,0,:,:],output_x[:,1,:,:], cplx=True)
        

        return x
