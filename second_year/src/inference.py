import sys, os

import util
import torch

import numpy as np
import random
import importlib

from tqdm import tqdm
from dataloader.data_loader import IITP_test_dataload
import pandas as pd
import soundfile as sf


class Hyparam_set():
    
    def __init__(self, args):
        self.args=args
    

    def set_torch_method(self,):
        try:
            torch.multiprocessing.set_start_method(self.args['hyparam']['torch_start_method'], force=False) # spawn
        except:
            torch.multiprocessing.set_start_method(self.args['hyparam']['torch_start_method'], force=True) # spawn
        

    def randomseed_init(self,):
        np.random.seed(self.args['hyparam']['randomseed'])
        random.seed(self.args['hyparam']['randomseed'])
        torch.manual_seed(self.args['hyparam']['randomseed'])
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args['hyparam']['randomseed'])

            device_primary_num=self.args['hyparam']['GPGPU']['device_ids'][0]
            device= 'cuda'+':'+str(device_primary_num)
        else:
            device= 'cpu'
        self.args['hyparam']['GPGPU']['device']=device
        return device
    def set_on(self):
        self.set_torch_method()
        self.device=self.randomseed_init()
       
        return self.args

class Learner_config():
    def __init__(self, args) -> None:
        self.args=args
    



    def memory_delete(self, *args):
        for a in args:
            del a

    def model_select(self):
        model_name=self.args['model']['name']
        model_import='models.'+model_name+'.main'

        
        model_dir=importlib.import_module(model_import)
        
        self.model=model_dir.get_model(self.args['model']).to(self.device)

        trained=torch.load(self.args['hyparam']['model'], map_location=self.device)
        self.model.load_state_dict(trained['model_state_dict'], )
        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])       
        

    def init_loss_func(self):
        
        

        if self.args['learner']['loss']['type']=='weighted_bce':
            from loss.bce_loss import weighted_binary_cross_entropy
            self.loss_func=weighted_binary_cross_entropy(**self.args['learner']['loss']['option'])
        elif self.args['learner']['loss']['type']=='BCEWithLogitsLoss':
            self.loss_func=torch.nn.modules.loss.BCEWithLogitsLoss(reduction='none')
            self.loss_func=torch.nn.modules.loss.BCELoss(reduction='none')

        self.loss_train_map_num=self.args['learner']['loss']['option']['train_map_num']
    
    def update(self, output, target):
       

        target=target[:, self.loss_train_map_num]
        output=output[:, self.loss_train_map_num].sigmoid()

        loss=self.loss_func(output, target)
    
        
        loss_mean=loss.mean()


        return loss_mean

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        self.model_select()
        self.init_loss_func()
        
        return self.args

class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.result_folder=self.args['hyparam']['result_folder']

        self.wav_save=self.args['hyparam']['wav_save']
        self.wav_folder=self.args['hyparam']['wav_folder']
        
        
        
       


    def save_output(self, DB_type):
        try:
            now_dict=self.save_config_dict[DB_type]
        except:
            now_dict=self.save_config_dict[int(DB_type)]
            DB_type=int(DB_type)
        
        with open(self.result_folder['inference_folder']+'/'+DB_type+'/result.txt', 'w') as f:
            save_folder=self.result_folder['inference_folder']+str(DB_type)+'/'
            
            pd.DataFrame(now_dict).to_csv(save_folder+'result.csv')
            f.write('\nSI-SDR\n\n')
            j=np.array(now_dict['SI-SDR']).mean()
            f.write(str(j))
            f.write('\n')


    def save_wav(self, DB_type, audio_list, iter_num):
        save_folder=self.result_folder['inference_folder']+str(DB_type)+'/wav/'+str(iter_num)+'/'
        os.makedirs(save_folder, exist_ok=True)
        sf.write(save_folder+'noisy.wav', audio_list[0].cpu().numpy(), 16000)
        
        sf.write(save_folder+'clean.wav', audio_list[1].cpu().numpy(), 16000)
        sf.write(save_folder+'estimate.wav', audio_list[2].cpu().numpy(), 16000)
        
    
    def error_update(self, DB_type, si_sdr,iter_num, audio_list):
        now_dict=self.save_config_dict[DB_type]
        now_dict['file_num'].append(iter_num)
        now_dict['SI-SDR'].append(si_sdr.cpu().item())
        self.save_config_dict[DB_type]=now_dict
        
        if self.wav_save:
            self.save_wav( DB_type, audio_list, iter_num)

        


    def config(self,):
        from copy import deepcopy

        self.save_config_dict=dict()

        metric_data={}
        metric_data['file_num']=[]
        metric_data['SI-SDR']=[]
        

        
        for room_type in self.result_folder['room_type']:
            os.makedirs(self.result_folder['inference_folder']+room_type, exist_ok=True)
            os.makedirs(self.result_folder['inference_folder']+room_type+'/wav', exist_ok=True)

            self.save_config_dict[room_type]=deepcopy(metric_data)
          

   

        return self.args

       



        

class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args

    
    def config(self):
        self.test_loader=IITP_test_dataload(self.args['dataloader']['test'])
        
       
        return self.args
        
        

class Tester():

    def __init__(self, args):


        self.args=args

        self.hyperparameter=Hyparam_set(self.args)
        self.args=self.hyperparameter.set_on()

        self.learner=Learner_config(self.args)
        self.args=self.learner.config()
        self.model=self.learner.model


        self.dataloader=Dataloader_config(self.args)
        self.args=self.dataloader.config()

        self.logger=Logger_config(self.args)
        self.args=self.logger.config()
        

    
    def run(self, ):
      
        
               
        
        self.test()   
        
        

    def test(self, ):
        self.model.eval()

        from asteroid.losses.sdr import SingleSrcNegSDR
        si_sdr_func=SingleSrcNegSDR('sisdr')
        

      
      
        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type=str(room_type)
            print(room_type)
            self.dataloader.test_loader.dataset.room_type=str(room_type)

            with torch.no_grad():
                for iter_num, (mixed, clean, SNR) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader), )):
        
                   
                    mixed=mixed.to(self.hyperparameter.device)
                    
                    clean=clean.to(self.hyperparameter.device)

                    out=self.model(mixed)
                    

                    si_sdr=-si_sdr_func(out, clean)
                   
                    self.logger.error_update(room_type, si_sdr, iter_num, [mixed[0,0], clean[0], out[0]])

                    

                    self.learner.memory_delete([mixed, clean, out, si_sdr])
   
                self.logger.save_output(room_type)




if __name__=='__main__':
    args=sys.argv[1:]
    
    args=util.util.get_yaml_args(args)

    t=Tester(args)
    t.run()