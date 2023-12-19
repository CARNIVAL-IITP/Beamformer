import sys, os

import util
import torch

import numpy as np
import random
import importlib

from tqdm import tqdm
from dataloader.data_loader import IITP_test_dataload

import pandas as pd

from asteroid.losses.sdr import SingleSrcNegSDR


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
        



    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        self.model_select()

        
        return self.args

class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.result_folder=self.args['hyparam']['result_folder']
        
        
        
       


    def save_output(self, DB_type):
        try:
            now_dict=self.save_config_dict[DB_type]
        except:
            now_dict=self.save_config_dict[int(DB_type)]
            DB_type=int(DB_type)
        
        with open(self.result_folder['inference_folder']+'/'+DB_type+'/result.txt', 'w') as f:

            f.write('si_sdr\n\n')
            k=(now_dict['si_sdr']/now_dict['num'])
            for j in k:
                    
                    j=str(j)            
                    f.write(j)
                    f.write('\n')
        df=pd.DataFrame(self.csv_dict[DB_type])
        df.to_csv(self.result_folder['inference_folder']+'/'+DB_type+'/result.csv')




    
    def error_update(self, DB_type, si_sdr, pkl_name, num=1):
        now_dict=self.save_config_dict[DB_type]
        
        now_dict['si_sdr']+=si_sdr
        now_dict['num']+=num

        self.save_config_dict[DB_type]=now_dict
        self.csv_dict[DB_type]['pkl_name'].append(pkl_name[0])
        self.csv_dict[DB_type]['si_sdr'].append(si_sdr.detach().cpu().numpy()[0])

   

    def config(self,):


        self.csv_dict=dict()

        self.save_config_dict=dict()

        metric_data={}
        metric_data['si_sdr']=0
        metric_data['num']=0
        self.pandas_df=dict()
        self.pandas_df['pkl_name']=[]
        self.pandas_df['si_sdr']=[]
 



        metric_data['number_of_degrees']=0
        os.makedirs('../results/', exist_ok=True)


   

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

        
        metric_func=SingleSrcNegSDR(reduction='none', zero_mean=True, take_log=True, sdr_type='sisdr')
        sisdr_list=[]

        mic_type=self.args['dataloader']['test']['mic_type']

        audio_save_dir='../results/circle_4_result/'
        os.makedirs(audio_save_dir, exist_ok=True)

        for room_type in tqdm(self.args['hyparam']['result_folder']['room_type'], desc='room, mic_type: '+mic_type):
            room_type=str(room_type)
            self.dataloader.test_loader.dataset.room_type=str(room_type)
            self.dataloader.test_loader.dataset.pkl_list=os.listdir(self.dataloader.test_loader.dataset.pkl_dir+room_type)
 
            with torch.no_grad():
                for iter_num, (mixed, target_wav, pkl_name) in enumerate (self.dataloader.test_loader):
     
                    

                   
                   
                    mixed=mixed.to(self.hyperparameter.device)
                    get_wav=target_wav.to(self.hyperparameter.device)

                    mixed=mixed.to(self.hyperparameter.device)
                    target_wav=target_wav.to(self.hyperparameter.device)

                    out, beamforming_weight_real, beamforming_weight_imag=self.model(mixed)
              

                    
           

                    si_sdr=-metric_func(out, target_wav)
                    si_sdr=si_sdr.detach().cpu().numpy()[0]
                    sisdr_list.append(si_sdr)

                    

                    self.learner.memory_delete([mixed, target_wav, out, si_sdr])
                
        df=pd.DataFrame(sisdr_list)





if __name__=='__main__':
    args=sys.argv[1:]
    
    args=util.util.get_yaml_args(args)

    t=Tester(args)
    t.run()