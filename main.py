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
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
    return data

def exp_mkdir(dir):
    now=datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir=dir+now+'/'
    os.makedirs(dir, exist_ok=True)
    f=open(dir+'description.txt', 'w')
    f.close()
    return dir

class trainer():
    def __init__(self,yaml_file, device):
        
        self.config_data=read_yaml(yaml_file)
        self.device=device
        self.train_loader=data_loader.Custom_dataload(self.config_data['train']['dataloader'])
        self.val_loader=data_loader.Custom_dataload(self.config_data['val']['dataloader'])
        self.model=model.FFT_CRN_IMAG(self.config_data['FFT']).to(self.device)
        self.optimizer=self.init_optimizer()
        self.loss_function=auraloss.time.SISDRLoss()
        self.exp_dir=exp_mkdir(self.config_data['exp']['result'])
        
        self.best_loss=484964
        self.scheduler=ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=4, verbose=True)

        for epoch in range(self.config_data['exp']['epoch']):
            self.train(epoch)
            self.val(epoch)
    
    
    def val(self, epoch):
        self.model=self.model.eval()
        losses = AverageMeter()
        times = AverageMeter()
        losses.reset()
        times.reset()

        with torch.no_grad():
            for iter_num, (input_data, label) in tqdm(enumerate(self.val_loader), desc='Validate', total=len(self.val_loader)):
                input_data=input_data.to(self.device)
            
                output=self.model(input_data)
                label=label.to(device)
                loss_iter=self.loss_function(output, label)
                losses.update(loss_iter.item())
                # break

        print('epoch %d, validate losses: %f'%(epoch, losses.avg), end='\r')  
        self.scheduler.step(losses.avg)
        if self.best_loss > losses.avg:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict_NS': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            if epoch>-1:
                torch.save(checkpoint, self.exp_dir + "/{}_model.tar".format(epoch))
            self.best_loss = losses.avg
        print("\n")
        # return losses.avg
        # exit()

    def train(self, epoch):
        self.model=self.model.train()
        self.optimizer.zero_grad()

        losses = AverageMeter()
        times = AverageMeter()
        losses.reset()
        times.reset()
        
        for iter_num, (input_data, label) in tqdm(enumerate(self.train_loader), desc='Train', total=len(self.train_loader)):
            input_data=input_data.to(self.device)
            
            output=self.model(input_data)
            label=label.to(device)
            loss_iter=self.loss_function(output, label)
            losses.update(loss_iter.item())
            
            loss_iter.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # break
        # print(losses.avg, times)
        print('epoch %d, training losses: %f'%(epoch, losses.avg), end='\r')  
        print("\n")
        # exit()
        # retur

    def init_optimizer(self):
        opti_option=self.config_data['train']['optimizer']
        opti_type=opti_option['type']
        if opti_type=='Adam':
            optimizer=torch.optim.Adam(self.model.parameters(), lr=float(opti_option['learning_rate']), weight_decay=float(opti_option['weight_decay']))
        
        return optimizer
        


if __name__=='__main__':
    device=randomseed_init(777)
    t=trainer('./config/train1.yaml', device)