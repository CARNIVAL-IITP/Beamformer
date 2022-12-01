import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utility import models


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3, 
                 kernel=3, num_spk=2, causal=True, ch_size=8, skip=False, condi_weight=[360, 128], condi_bias=[360, 128], Film_loc=8, padding='auto'):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk
        self.skip=skip

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal
        self.ch_size=ch_size
        self.Film_loc=Film_loc
        self.padding=padding
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # conditioning FiLM
        self.FiLM_weight=nn.Conv1d(condi_weight[0], condi_weight[1], 1)
        self.FiLM_bias=nn.Conv1d(condi_bias[0], condi_bias[1], 1)

        # TCN separator
        self.TCN = models.TCN(self.ch_size, self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                              self.layer, self.stack, self.kernel, causal=self.causal, skip=self.skip, Film_loc=self.Film_loc)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
     
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        # print(input.shape, pad_aux.shape)
        # exit()

        return input, rest
        
    def forward(self, input, ssl_condition):
        
        batch_size, ch_size, sample_size= input.shape

        ref_ch=[ch_size*k for k in range(batch_size)]
        input=input.view(batch_size*ch_size, sample_size)
        # padding
        if self.padding =='auto':
            output, rest = self.pad_signal(input)
        else:
            output=F.pad(input, self.padding, mode='constant').unsqueeze(1)
       
        
        
        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L
        # print(enc_output.shape)
        # exit()
        target_output=enc_output[ref_ch]
        
        enc_output=enc_output.view(batch_size, -1, enc_output.shape[-1])
        # print(enc_output.shape)
        # exit()
        
        remainder=enc_output.shape[-1]%ssl_condition.shape[-1]
        
        
        frame_ratio=enc_output.shape[-1]//ssl_condition.shape[-1]
        
        ssl_condition=ssl_condition.squeeze(1).repeat_interleave(frame_ratio, dim=-1)
        # ssl_condition=F.pad(ssl_condition, [0, remainder], mode='constant')
        # print(ssl_condition.shape, enc_output.shape)
        # exit()
        # print(ssl_condition.shape)
        # exit()
        ssl_weight=self.FiLM_weight(ssl_condition)
        ssl_bias=self.FiLM_bias(ssl_condition)
        

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output, ssl_weight, ssl_bias)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = target_output.unsqueeze(1) * masks  # B, C, N, L
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        if self.padding=='auto':
            output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        else:
            output=output[:, :, self.padding[0]:-self.padding[1]]
        output = output.view(batch_size, -1)  # B, C, T
        # print(output.shape)
        # exit()
    
        return output

def test_conv_tasnet():
    x = torch.rand(2, 32000)
    nnet = TasNet()
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()