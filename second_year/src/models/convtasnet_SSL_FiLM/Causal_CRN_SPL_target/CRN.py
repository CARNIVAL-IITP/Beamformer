from torch.nn.modules import conv
from .FFT import EPSILON, ConvSTFT, ConviSTFT
from torch import nn
import torch
from util import *
import matplotlib.pyplot as plt
import numpy as np

class Causal_Conv2D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Causal_Conv2D_Block, self).__init__()
        
        
        self.conv2d=nn.Conv2d(*args, **kwargs)


        self.norm=nn.BatchNorm2d(args[1])
        
        self.activation=nn.ELU()

    def forward(self, x):
        original_frame_num=x.shape[-1]
        x=self.conv2d(x)
        x=self.norm(x)
        x=self.activation(x)   
        x=x[...,:original_frame_num]    
        
       

        return x

class Conv1D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv1D_Block, self).__init__()
        
        
        self.conv1d=nn.Conv1d(*args, **kwargs)


        self.norm=nn.BatchNorm1d(args[1])
        
        self.activation=nn.ELU()

    def forward(self, x):
        
        x=self.conv1d(x)
        x=self.norm(x)
        x=self.activation(x)       
        
       

        return x


class crn(nn.Module):
    def __init__(self, config, output_num, azi_size):
        super(crn, self).__init__()

        
        self.output_num=output_num
        self.azi_size=azi_size

        
        self.cnn_num=config['CNN']['layer_num']
        self.kernel_size=config['CNN']['kernel_size']
        self.filter_size=config['CNN']['filter']

        self.max_pool_kernel=config['CNN']['max_pool']['kernel_size']
        self.max_pool_stride=config['CNN']['max_pool']['stride']

        args=[2*(config['input_audio_channel']-1),self.filter_size,self.kernel_size] # in_channel, out_channel, kernel size
       
        kwargs={'stride': 1, 'padding': [1,2], 'dilation': 1}

      


        self.cnn=nn.ModuleList()
        self.pooling=nn.ModuleList()
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))
        self.pooling.append(nn.MaxPool2d(self.max_pool_kernel, stride=self.max_pool_stride))

        args[0]=config['CNN']['filter']       
        for count in range(self.cnn_num-1):
            self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))
            self.pooling.append(nn.MaxPool2d(self.max_pool_kernel, stride=self.max_pool_stride))
    
        self.GRU_layer=nn.GRU(**config['GRU'])

        self.azi_mapping_conv_layer=nn.ModuleList()
        self.azi_mapping_final=nn.ModuleList()

        args[0]=config['GRU']['hidden_size']
        args[1]=config['GRU']['hidden_size']
        args[2]=1
        kwargs['padding']=0
      
        for _ in range(output_num):
            self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))
            self.azi_mapping_final.append(nn.Conv1d(config['GRU']['hidden_size'], self.azi_size, 1))
     
        

        
        

    def forward(self, x):
      
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):
            # print(x.shape)
            x=cnn_layer(x)[...,:x.shape[-1]]
            x=pooling_layer(x)
            # print(x.shape)
            # exit()
        
        
        b, c, f, t=x.shape
        x=x.view(b, -1, t).permute(0,2,1)

        # x_cnn=self.lstm_cnn_layer(x)
        self.GRU_layer.flatten_parameters()
        x, h=self.GRU_layer(x)
        x=x.permute(0,2,1)
        
        outputs=[]

        for final_layer, cnn_layer in zip(self.azi_mapping_final, self.azi_mapping_conv_layer):
            x=cnn_layer(x)
            res_output=final_layer(x)
            outputs.append(res_output)
        output=torch.stack(outputs).permute(1,0,2,3)
        
        return output


class main_model(nn.Module):
    def __init__(self, config):
        super(main_model, self).__init__()
        self.config=config
        
        self.eps=np.finfo(np.float32).eps
        self.ref_ch=self.config['ref_ch']

        ###### sigma

        self.sigma=torch.tensor(self.config['sigma_start'])
        self.sigma_max=torch.tensor(self.config['sigma_end']['max'])
        self.sigma_min=torch.tensor(self.config['sigma_end']['min'])
        self.sigma_rate=torch.tensor(self.config['sigma_rate'])
        self.sigma_udpate_method=self.config['sigma_update_method']
        
        self.iteration_count=0        
        self.epoch_count=0
        self.now_epoch=0

       
        ######
       
        self.max_spk=self.config['max_spk']
        self.degree_resolution = self.config['degree_resolution']
        self.azi_size=360//self.degree_resolution

        self.stft_model=ConvSTFT(**self.config['FFT'])
        self.crn=crn(self.config['CRN'], self.sigma.shape[0], self.azi_size)
    
    def save_fig(self, data):
        data=data[2].cpu().detach().numpy()
        fig=plt.figure(figsize=(7,3),)#, nrows=azi_num, ncols=1, sharey=True)

        for i in range(data.shape[1]):
            plt.subplot(1, data.shape[1], i+1)

            lj=plt.imshow(data[:,i],  vmin=0, vmax=1.0, cmap = plt.get_cmap('plasma'), aspect='auto')
            plt.colorbar(lj)

        plt.tight_layout()
        plt.savefig('./png_folder/target_coding.png', dpi=400, interpolation="none")
        plt.close()
        plt.clf()
        plt.cla()
        exit()

    def sigma_update(self, iter_num, epoch):
        if iter_num%500==0:
                print(self.sigma)
        def update():
            

            if self.sigma_udpate_method=='add':
                self.sigma+=self.sigma_rate
            elif self.sigma_udpate_method=='multiply':
                self.sigma*=self.sigma_rate
            else:
                "Not exist!!!"
                exit()

            self.sigma=torch.clamp(self.sigma, self.sigma_min, self.sigma_max)

       
        if self.training:

            if self.config['iter']['update']:
                if self.iteration_count!=self.config['iter']['update_period']:
                    self.iteration_count+=1
                
                else:
                    print('sigma_iter update')
                    update()
                    self.iteration_count=0
                    return
            
            if self.config['epoch']['update']:

                if self.now_epoch!=epoch:
                    self.now_epoch=epoch
                    self.epoch_count+=1
                
                if self.epoch_count==self.config['epoch']['update_period']:
                    print('sigma_epoch update')
                    update()
                    self.epoch_count=0
                    return 

    


    def make_target(self, target, azi, iter_num, epoch):
        
        

        azi_target=torch.div(azi, 360//self.azi_size, rounding_mode='floor').long()        
        azi_range=torch.arange(0, self.azi_size).unsqueeze(0).to(azi_target.device)

        # target_for_loss=torch.zeros((target.shape[0], self.azi_size,self.sigma_resolution_tensor.shape[0], target.shape[-1]), dtype=torch.float32, device=azi_target.device)
        
        distance=azi_target.unsqueeze(-1)*self.degree_resolution-azi_range*self.degree_resolution
        
        distance_abs=torch.abs(distance)
        distance_abs=torch.stack((distance_abs, 360-distance_abs), dim=0)
        distance=torch.pow(torch.min(distance_abs, dim=0).values,2)

        
        sigma=self.sigma.view(1,1,-1).to(distance.device)

        labelling=torch.exp(-distance.unsqueeze(-1)/sigma**2).unsqueeze(-1)

        target=target.unsqueeze(2).unsqueeze(2)
        target=labelling*target
        target=torch.max(target, dim=1).values
        # print(target.shape)
        # exit()

        target=target.permute(0,2,1,3) 

        # tt=target[0, :, 0, :].cpu().detach().numpy()
        # plt.imshow(tt)
        # plt.savefig('../data/target.png')
        # plt.close()

        # plt.plot(tt[:,5])
        # plt.savefig('../data/one_frame.png')

        self.sigma_update(iter_num, epoch)
       
        return target # batch, sigma_num, degree, frame

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
    
    def vad_framing(self, vad_batch):

        vad_output_th = vad_batch.mean(axis=2) > 2 / 3
        
        vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis, np.newaxis]
        vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(maps.device)
        repeat_factor = np.array(maps.shape)
        repeat_factor[:-2] = 1
        maps *= vad_output_th.float().repeat(repeat_factor.tolist())
    
    def plot_vad_freq(self, data):
        data=data[0].detach().cpu().numpy()

        for i in range(data.shape[0]):
            plt.plot(data[i])
        plt.savefig('../results/vad_freq.png')
        exit()
    
    def forward(self, x, vad, each_azi, iter_num, epoch, LOCATA=False):

        ###### irtf feature
        x, vad_frame=self.irtf_featue(x, vad)
        
        # self.vad_framing(vad)
        
        

        if LOCATA:
            target=self.stft_model.azimuth_strided(vad_frame, each_azi).unsqueeze(0)
        else:
            target=self.make_target( vad_frame, each_azi, iter_num, epoch)
            # target=self.cjh_target(vad_frame, each_azi, iter_num, epoch)

       
        
       
        
        x=self.crn(x)
        
        
        
     
      
        
        return x, target, vad_frame

if __name__=='__main__':
    device='cuda'
    yaml_file=load_yaml('./config/train.yaml')['model']['structure']

    

    
    model=main_model(yaml_file).eval().to(device)

    batch=2
    length=64000
    with torch.no_grad():
        for i in range(1):
            mixture=torch.randn((batch, 8, length)).to(device)
            target=torch.randn((batch, 2, length)).to(device)
            azi=torch.tensor([180, 355]).to(device).unsqueeze(0)
            azi=azi.repeat_interleave(batch, dim=0)
            
            output=model(mixture, target, azi)

