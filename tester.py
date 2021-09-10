import torch
import numpy as np
import random
import yaml
import data_loader
from datetime import datetime
from torch.utils.data import DataLoader 
import os
import model
import auraloss
from tqdm import tqdm
import soundfile as sf
import torch.nn.functional as F

def randomseed_init(num):
    np.random.seed(num)
    random.seed(num)
    torch.manual_seed(num)
    # torch.Generator.manual_seed(num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(num)
        return 'cuda'
    else:
        return 'cpu'

def read_yaml(yaml_file):
    yaml_file=open(yaml_file, 'r')
    data=yaml.safe_load(yaml_file)
    return data['test']

def calc_snr(inference, label, noise):
    # print(noise.shape)
    # exit()
    noise=torch.norm(noise, 1)
    loss_function=auraloss.time.SISDRLoss()
    res=-loss_function(inference, label)
    return res.cpu()
    # print(res)
    # exit()

    # # noise=torch.abs(noise).sum(dim=1)
    # # print(noise)
    # # exit()
    # # label=label[:,inference.shape[1]]
    # residual=torch.norm(inference-label, 1)
    # # residual=torch.abs(inference-label).sum(dim=1)
    # return 20*torch.log10(noise/residual)

class tester():
    def __init__(self,yaml_file, device):
        self.config_data=read_yaml(yaml_file)
        self.device=device
        
        self.test_loader=data_loader.Custom_dataload(self.config_data['dataloader'])
        
        self.model=model.FFT_CRN_IMAG(self.config_data['FFT']).to(self.device).eval()
        self.model.load_state_dict(torch.load(self.config_data['model']['trained'])['state_dict_NS'])
        self.snr_list=[]
        

        with torch.no_grad():
            for iter_num, (input_data, label, snr, file_name) in tqdm(enumerate(self.test_loader), desc='Test', total=len(self.test_loader)):
                # print(input_data.shape)
                # exit()
                file_name=file_name[0].split('/')[-1]
                input_data=input_data.to(self.device)            
                output=self.model(input_data)
                label=label.to(device)
                margin=-output.shape[1]+label.shape[1]
                
                
                output=F.pad(output, pad=(0, margin), mode='constant', value=0)
                # print(output.shape, label.shape)
                # print(snr)
                snri=calc_snr(output, label, input_data[:,0,:]-label)
                # print(snri-snr)
                # exit()
                # exit()
                output=output.cpu().squeeze().numpy()
                sf.write('/home/intern0/Desktop/project/IITP/Beamformer/exp_result/2021_09_08_20_59_15/wav/'+file_name, output, 16000)
                # print(snri)
                # exit()
                self.snr_list.append(snri.cpu())
                # if iter_num==3:
                #     break
        self.snr_list=torch.tensor(self.snr_list).mean()
        print(self.snr_list)
        exit()
                
                


if __name__=='__main__':
    device=randomseed_init(777)
    t=tester('./config/train1.yaml', device)