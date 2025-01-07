import sys, os
import util
import torch
import numpy as np
import random
import importlib
import math
import wandb
from tqdm import tqdm
from dataloader.data_loader import Train_dataload, Eval_dataload
import matplotlib.pyplot as plt
import pandas as pd
# from dataloader.mic_array import get_ULA
from scipy.spatial.transform import Rotation as Rotation_class
import gc

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
        # torch.Generator.manual_seed(self.args['randomseed'])
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
        self.args_learner=self.args['learner']
        self.beampattern_device=None
        

    def get_steering_vector_from_target_direction_vector(self, target_direction_vectors):
        # print(target_direction_vectors.shape)
        # exit(1)
        self.array_pos=self.array_pos.astype(np.float32)
        position_vector=self.array_pos[0:1]-self.array_pos
        position_vector=torch.from_numpy(position_vector).to(target_direction_vectors.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1, 1, 1, M, 3

        target_direction_vectors=target_direction_vectors.unsqueeze(-2) # b,Z, Z, 1, 3

        tau=torch.sum(position_vector*target_direction_vectors, dim=-1, keepdim=True) / self.sound_speed # b, Z, Z, M, 1

        freq_bins=torch.from_numpy(self.freq_bins).to(target_direction_vectors.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1, 1, 1, F

        # print(freq_bins.shape, tau.shape)
        steering_vector=torch.exp(-1j*2.0*torch.pi*freq_bins*tau) # b, Z, Z, M, F
        
        return steering_vector
    
    def get_steering_vector_from_rotated(self, target_direction_vectors):

        # steering_vector=torch.zeros((target_direction_vectors.shape[0],target_direction_vectors.shape[1], target_direction_vectors.shape[2],  self.mic_num, self.freq_bins.shape[0]), dtype=torch.complex64, device=self.beampattern_device) # b, Z, Z, M, F

       
        # azimuth=torch.arctan(target_direction_vectors[..., 0]/target_direction_vectors[..., 1])
        # elevation=torch.arccos(target_direction_vectors[..., 2])

        steering_vector=self.get_steering_vector_from_target_direction_vector(target_direction_vectors) # b, Z, Z, M, F
        return steering_vector
        # self.direction_vector=self.get_direction_vector()

    def beampattern_init(self, array_pos):
        self.array_pos=array_pos
        # self.data
        

        self.mic_num=self.array_pos.shape[0]

        self.freq_bins=np.linspace(0, 1, self.args_learner['beampattern']['fft_len']//2+1, dtype=np.float32)*self.args_learner['beampattern']['fs']/2
        
        
        if self.args_learner['beampattern']['device']=='cpu':
            self.beampattern_device='cpu'
        else:
            self.beampattern_device=self.device
       
        # self.freq_bins=np.linspace(0, 1, self.args_learner['beampattern']['fft_len']//2+1, dtype=np.float32)*self.args_learner['beampattern']['fs']/2


        self.sound_speed=self.args_learner['beampattern']['sound_speed']

        self.angle_candidates=self.args_learner['beampattern']['angle_candidates']
        self.angle_candidates_num=len(self.angle_candidates)

        self.sigma=torch.tensor(self.args_learner['beampattern']['sigma'])
        self.p=torch.tensor(self.args_learner['beampattern']['p'])

        self.target_tensor, self.target_direction_vectors=self.get_target_and_steering_vectors()

        self.target_tensor=self.target_tensor.astype(np.float32)
        self.target_direction_vectors=self.target_direction_vectors.astype(np.float32)

        # print(self.target_tensor.shape)
        # print(self.target_tensor)
        # exit(1)

        self.azimuth_bins=np.arange(0, 360, self.args_learner['beampattern']['theta_step'], dtype=np.float32)
        self.azimuth_bins=np.expand_dims(self.azimuth_bins, axis=-1)

        self.elevation_bins=np.arange(0, 181, self.args_learner['beampattern']['theta_step'], dtype=np.float32)

        self.elevation_bins=np.expand_dims(self.elevation_bins, axis=0)

        # print(self.target_tensor.shape)
        # print(self.target_direction_vectors.shape)
        # exit(1)
        # plt.figure()
        # plt.imshow(self.target_tensor, aspect='auto', origin='lower', vmin=0.0, vmax=1.0)
        # plt.colorbar()
        # plt.savefig('../results/pngs/target_tensor.png')
        # plt.close()
        # exit(1)
    def sph2cart(self, azimuth, elevation, r=1.0):
        x = r * np.cos(azimuth) * np.sin(elevation)
        y = r * np.sin(azimuth) * np.sin(elevation)
        z = r * np.cos(elevation)
        # print(z)
        
        return x, y, z
    def soft_labelling(self, angle):
        sigma=torch.deg2rad(self.sigma)
        kappa_d=torch.log(self.p)/(torch.cos(sigma)-1)

        

        angle_label=torch.exp(kappa_d*(torch.cos(torch.deg2rad(torch.tensor(angle)))-1))

        
        return angle_label.item()

    def direction_vector(self, azimuth,elevation):
        x,y,z=self.sph2cart(azimuth, elevation, r=1.0)
        return np.array([x,y,z])   

    def direction_vector_torch(self, azimuth,elevation):
        x,y,z=self.sph2cart_torch(azimuth, elevation, r=1.0)

        print(x.shape,y.shape, z.shape)
        exit(1)
        return torch.tensor([x,y,z]) 
     
    def get_target_and_steering_vectors(self):

        square_size=2*self.angle_candidates_num-1
        
        target_tensor=np.zeros((square_size, square_size), dtype=np.float32)
        direction_vector=np.zeros((square_size, square_size, 3), dtype=np.float32)

        row=self.angle_candidates_num-1
        col=self.angle_candidates_num-1
        

      

        for p in range(1, self.angle_candidates_num+1):
         
            target_value=self.soft_labelling(self.angle_candidates[p-1])
            u_zero=self.direction_vector(0.0, np.deg2rad(self.angle_candidates[p-1]))

           

            if p==1:
                target_tensor[col, row]=target_value
                direction_vector[col, row]=u_zero
                # continue

            else:
            

                for q in range(1, 8*(p-1)+1):
                    # print(q, p)
                    # print(col, row)
                    # qeqe=input()
                    # if qeqe=='q':
                    #     exit(1)

                 
                    
                    rotation_function=Rotation_class.from_euler('z', 360*(q-1)/(8*p), degrees=True)
                    u=rotation_function.apply(u_zero)

                    # degree=np.dot([0,0, 1], u)
                    # degree=np.rad2deg(np.arccos(degree))
                    # print(degree)
                    
                    target_tensor[col, row]=target_value
                    direction_vector[col, row]=u

                    if q<p or q >= (7*p-6):
                        row-=1
                    elif q<(3*p-2):
                        col-=1
                    elif q<(5*p-4):
                        row+=1
                    elif q<(7*p-6):
                        col+=1

                
            col+=1

        return target_tensor, direction_vector
    
    def memory_delete(self, *args):
        for a in args:
            del a

    def model_select(self):
        model_name=self.args['model']['name']
        model_import='models.'+model_name+'.main'

        
        model_dir=importlib.import_module(model_import)
        
        self.model=model_dir.get_model(self.args['model']).to(self.device)
        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])       
        

    def init_optimizer(self):
        
        a=importlib.import_module('torch.optim')
        assert hasattr(a, self.args['learner']['optimizer']['type']), "optimizer {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer']['type'])
     
        self.optimizer=a(self.model.parameters(), **self.args['learner']['optimizer']['config'])
        self.gradient_clip=self.args['learner']['optimizer']['gradient_clip']
    
        
    def init_optimzer_scheduler(self, ):
        a=importlib.import_module('torch.optim.lr_scheduler')
        assert hasattr(a, self.args['learner']['optimizer_scheduler']['type']), "optimizer scheduler {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer_scheduler']['type'])

        self.optimizer_scheduler=a(self.optimizer, **self.args['learner']['optimizer_scheduler']['config'])

    def sph2cart_torch(self, azimuth, elevation, r=1.0):

        x=r*torch.cos(azimuth)*torch.sin(elevation)
        y=r*torch.sin(azimuth)*torch.sin(elevation)
        z=r*torch.cos(elevation)

        # print(azimuth, elevation)
        # print(torch.cos(elevation))
        # exit()
        # print(x,y,z)
        # exit(1)

        return x, y, z
    
    def new_get_target_pattern(self, target_azimuth, target_elevation):

        # print(target_azimuth, target_elevation)

        # target_azimuth=target_azimuth*0  +90
        # target_elevation=target_elevation*0 + 90
        # # print(target_azimuth-target_azimuth_temp)
        # # exit()

        # print(target_azimuth, target_elevation)
        
        target_azimuth=torch.deg2rad(target_azimuth).to(self.beampattern_device).detach().view(-1, 1,1) #*0 + 90
        target_elevation=torch.deg2rad(target_elevation).to(self.beampattern_device).detach().view(-1, 1,1) #*0 + 90   

        

        x,y,z=self.sph2cart_torch(target_azimuth, target_elevation, r=1.0)

        target_direction=torch.stack([x,y,z], dim=1)

        # print(target_direction)
        # print(target_azimuth, target_elevation)
        
        print(target_direction.shape)
        exit(1)

        direction_vector=self.direction_vector_torch(target_azimuth, target_elevation).to(self.beampattern_device).unsqueeze(0)

        print(direction_vector)
        exit(1)
        # print(target_direction.shape)
        # print(direction_vector.shape)
        # exit(1)

        cos=torch.sum(target_direction*direction_vector, dim=1)
        cos=torch.clamp(cos, min=-1.0, max=1.0)

        # plt.subplot(3,1,1)

        # plt.imshow(cos[0].cpu().detach().numpy().T, aspect='auto', origin='lower')
        # plt.colorbar()
        
        angle_distance=torch.acos(cos)
        # print(angle_distance.min(), angle_distance.max())
        # angle_distance=torch.rad2deg(angle_distance)

        # plt.subplot(3,1,2)
        # plt.imshow(torch.rad2deg(angle_distance)[0].cpu().detach().numpy().T, aspect='auto', origin='lower')
        # plt.colorbar()
        
        sigma=torch.deg2rad(self.sigma)
        kappa_d=torch.log(self.p)/(torch.cos(sigma)-1)

        angle_label=torch.exp(kappa_d*(torch.cos(angle_distance)-1))

        # plt.subplot(3,1,3)
        # plt.imshow(angle_label[0].cpu().detach().numpy().T, aspect='auto', origin='lower')
        # plt.colorbar()

        # plt.tight_layout()
        # plt.savefig('../results/beam_sample/cos.png')
        # exit(1)
        
        return angle_label
    def get_beampattern_new(self, beamforming_weight, steering_vectors):
        beamforming_weight=beamforming_weight.permute(0,1,3,2).unsqueeze(2).unsqueeze(2) # b, Z, M, F, 1, 1
        steering_vectors=steering_vectors.unsqueeze(1) # b, 1, 1, M, F, 1
        beampattern=torch.sum(steering_vectors.conj()*beamforming_weight, dim=-2) # b, T, Z, Z, F
        
        target_direction_value=beampattern[:,:, self.angle_candidates_num-1, self.angle_candidates_num-1, :]

        beampattern_power=torch.abs(beampattern)**2

        return beampattern_power, target_direction_value
        
    def get_pattern(self,beamforming_weight, target_azimuth, target_elevation, rotation_matrix):

        self.beampattern_device=beamforming_weight.device
        rotation_matrix=rotation_matrix.to(self.beampattern_device)

        target_direction_vectors=torch.from_numpy(self.target_direction_vectors).to(self.beampattern_device)

        

        batch_size=beamforming_weight.shape[0]
        target_direction_vectors=target_direction_vectors.unsqueeze(0).repeat_interleave(batch_size, dim=0)

        rotated_target_direction_vectors=torch.einsum('bij,bnmj->bnmi', rotation_matrix, target_direction_vectors).type_as(rotation_matrix)

        

        target_steering_vectors=self.get_steering_vector_from_rotated(rotated_target_direction_vectors)

        # print(target_steering_vectors.dtype)
        # exit(1)

        beampattern_power, target_direction_value=self.get_beampattern_new(beamforming_weight, target_steering_vectors)

        

        # target_theta=self.get_theta(azimuth, elevation)

        # target_pattern=self.get_target_patthern(target_azimuth, target_elevation)
        target_tensor=torch.from_numpy(self.target_tensor).to(self.beampattern_device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat_interleave(batch_size, dim=0).repeat_interleave(self.freq_bins.shape[0], dim=-1).repeat_interleave(beampattern_power.shape[1], dim=1)


        return beampattern_power, target_tensor, target_direction_value


    def init_loss_func(self):
        
        

        if self.args['learner']['loss']['type']=='weighted_bce':
            from loss.bce_loss import weighted_binary_cross_entropy
            self.loss_func=weighted_binary_cross_entropy(**self.args['learner']['loss']['option'])
        elif self.args['learner']['loss']['type']=='BCEWithLogitsLoss':
            self.loss_func=torch.nn.modules.loss.BCEWithLogitsLoss(reduction='none')
            self.loss_func=torch.nn.modules.loss.BCELoss(reduction='none')


        elif self.args['learner']['loss']['type']=='kld':
            self.loss_func=torch.nn.modules.loss.KLDivLoss(reduction='none')
        elif self.args['learner']['loss']['type']=='mse':
            self.loss_func=torch.nn.modules.loss.MSELoss(reduction='none')
        elif self.args['learner']['loss']['type']=='sync_SI_SDR':
            from loss import SI_SDR_sync
            self.loss_func=SI_SDR_sync.sync_SI_SDR(reduction='none')
        self.bce_loss=torch.nn.modules.loss.BCELoss(reduction='none')
        self.mse_loss=torch.nn.modules.loss.MSELoss(reduction='none')

        

        self.loss_train_map_num=self.args['learner']['loss']['option']['train_map_num']
        self.loss_weight=self.args['learner']['loss']['option']['each_layer_weight']

        if self.args['learner']['loss']['optimize_method']=='min':
            self.best_val_loss=math.inf
            self.best_train_loss=math.inf
        else:
            self.best_val_loss=-math.inf
            self.best_train_loss=-math.inf

    def train_update(self, output, target, beampattern_power, target_tensor):

        
            
        # target=target[:, self.loss_train_map_num]
        # output=output[:, self.loss_train_map_num].sigmoid()
        # print(self.loss_func)
        # exit(1)
        loss=self.loss_func(output, target)

        mse_loss=self.mse_loss(beampattern_power, target_tensor)
        # bce_loss=self.bce_loss(output, target)

        


        # for j in range(len(self.loss_weight)):
        #     loss[:, j]=loss[:,j]*self.loss_weight[j]



        loss_mean=loss.mean()+0.1*mse_loss.mean()

        del loss, mse_loss

        # print(loss.mean(), mse_loss.mean())

        if torch.isnan(loss_mean):
            print('nan occured')
            self.optimizer.zero_grad()
            return loss_mean

        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

       

        return loss_mean

    def test_update(self, output, target):
       
        # print(output.shape, self.loss_train_map_num)
        # exit()
        loss=self.loss_func(output, target)

        # for j in range(len(self.loss_weight)):
        #     loss[:, j]=loss[:,j]*self.loss_weight[j]
        loss_mean=loss.mean()
        

        if torch.isnan(loss_mean):
            print('nan occured')
            self.optimizer.zero_grad()
            return loss_mean

        return loss_mean

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        self.model_select()
        self.init_optimizer()
        self.init_optimzer_scheduler()
        self.init_loss_func()
        return self.args

class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.csv=dict()
        self.csv['train_epoch_loss']=[]
        self.csv['train_best_loss']=[]
        self.csv['test_epoch_loss']=[]
        self.csv['test_best_loss']=[]

        self.csv_dir=self.args['logger']['save_csv']
        self.model_save_dir=self.args['logger']['model_save_dir']
        self.png_dir=self.args['logger']['png_dir']

        if self.args['logger']['optimize_method']=='min':
            self.best_test_loss=math.inf
            self.best_train_loss=math.inf
        else:
            self.best_test_loss=-math.inf
            self.best_train_loss=-math.inf

    def train_iter_log(self, loss):
        try:
            wandb.log({'train_iter_loss':loss})
        except:
            None
        self.epoch_train_loss.append(loss.cpu().detach().item())

       
    def plotting_target_output(self, iter_num, out, target):
        out=out.sigmoid().detach().cpu().numpy()[0]
        target=target.detach().cpu().numpy()[0]
        azi_resolution=1
        
        #### plotting
        if iter_num%200==0:
            # print(self.loss_function.weights[1])
            
            fig=plt.figure(figsize=(7,7),)#, nrows=azi_num, ncols=1, sharey=True)

            # for (row, big_ax), azi  in zip(enumerate(big_axes, start=1),output_dict.keys()):
            #     big_ax.set_title(azi, fontsize=16)
            
          
            
            for num in range(out.shape[0]):
                
                output_now=out[num]
                target_now=target[num]

                

                
                # lj=plt.imshow(target_now,  vmin=0, vmax=1.0, cmap = plt.get_cmap('plasma'), aspect='auto')
                # plt.colorbar(lj)
                # plt.savefig('../results/estimate/png_sample.png')
                # exit()
                
                
              

                ytick_front=[0,output_now.shape[0]//2, output_now.shape[0]-1]
                ytick_back=[0,(output_now.shape[0]//2)*azi_resolution, (output_now.shape[0]-1)*azi_resolution]

                plt.subplot(out.shape[0],3,num*3+1)
                
                
                lj=plt.imshow(output_now,  vmin=0, vmax=1.0, cmap = plt.get_cmap('plasma'), aspect='auto')
                plt.colorbar(lj)
                
                plt.yticks(ytick_front, ytick_back)
                # plt.title('{} Output'.format(azi))
                
                # plt.imshow(target_now)
                # plt.savefig('../results/estimate/png_sample.png')
                # exit()
                plt.subplot(out.shape[0],3,num*3+2)
                

                output_gap=target_now-output_now
                lj=plt.imshow(output_gap,vmin=-1.0, vmax=1.0,  cmap = plt.get_cmap('seismic'), aspect='auto')
                plt.colorbar(lj)
                plt.yticks(ytick_front, ytick_back)
                


                plt.subplot(out.shape[0],3,num*3+3)
                
                
                tk=plt.imshow(target_now , vmin=0, vmax=1, cmap = plt.get_cmap('plasma'),  aspect='auto')
                plt.colorbar(tk)
                plt.yticks(ytick_front, ytick_back)
                
            png_name='../results/estimate/estimaste_{}.png'.format(iter_num)
            os.makedirs(os.path.dirname(png_name), exist_ok=True)
            plt.tight_layout()
            plt.savefig(png_name, dpi=400,)
            plt.close()
            plt.clf()
            plt.cla()
      
            
        
        return 

    def train_epoch_log(self):
        loss_mean=np.array(self.epoch_train_loss).mean()

        self.csv['train_epoch_loss'].append(loss_mean)

        if self.best_train_loss > loss_mean:
            self.best_train_loss = loss_mean 

        try:
            wandb.log({'train_epoch_loss':loss_mean})
            wandb.log({'train_best_loss':self.best_train_loss})
        except:
            None

        self.csv['train_best_loss'].append(self.best_train_loss)



    def test_iter_log(self, loss):
        try:
            wandb.log({'test_iter_loss':loss})
        except:
            None
        self.epoch_test_loss.append(loss.cpu().detach().item())

    def test_epoch_log(self, optimizer_scheduler):
        loss_mean=np.array(self.epoch_test_loss).mean()
        self.csv['test_epoch_loss'].append(loss_mean)

        self.model_save=False
        if self.best_test_loss > loss_mean:
            self.model_save=True
            self.best_test_loss = loss_mean 
        try:
            wandb.log({'test_epoch_loss':loss_mean})
            wandb.log({'test_best_loss':self.best_test_loss})
        except:
            None
        self.csv['test_best_loss'].append(self.best_test_loss)

        optimizer_scheduler.step(loss_mean)
        

        

    def epoch_init(self,):
        self.epoch_train_loss=[]
        self.epoch_test_loss=[]
    

    def epoch_finish(self, epoch, model, optimizer):
        os.makedirs(os.path.dirname(self.csv_dir), exist_ok=True)
        pd.DataFrame(self.csv).to_csv(self.csv_dir)

        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }

        os.makedirs(os.path.dirname(self.model_save_dir + "best_model.tar"), exist_ok=True)
        if self.model_save:
            os.makedirs(os.path.dirname(self.model_save_dir + "best_model.tar"), exist_ok=True)
            torch.save(checkpoint, self.model_save_dir + "best_model.tar")
            print("new best model\n")
        torch.save(checkpoint,  self.model_save_dir + "{}_model.tar".format(epoch))

        torch.save(checkpoint,  self.model_save_dir + "last_model.tar")

        
        util.util.draw_result_pic(self.png_dir, epoch, self.csv['train_epoch_loss'],  self.csv['test_epoch_loss'])



    def wandb_config(self):
        if self.args['logger']['wandb']['wandb_ok']:
            wandb.init(**self.args['logger']['wandb']['init'])      
        return self.args  
  
    def config(self,):
        self.wandb_config()
        return self.args
        

class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args
        
    
    def config(self):
        self.test_loader=Eval_dataload(self.args['dataloader']['test'])
        self.train_loader=Train_dataload(self.args['dataloader']['train'], self.args['hyparam']['randomseed'])     
        # exit()
        return self.args   
        
        
        

class Trainer():
    def temp(self):
        dir='/root/harddisk1/Dataset/librispeech/ICASPP_2023_DOA_test_set/'
        pkl=os.listdir(dir)
        dictdcit=dict()
        dictdcit['data']=pkl
        save_dict='/root/share/share/ICASSP_2023_DOA_deepsupervision_curri/src/metadata/test_csv.csv'
        pd.DataFrame(dictdcit).to_csv(save_dict)
        print(len(pkl))
        exit()

    def __init__(self, args):

        # self.temp()
        self.args=args

        self.hyperparameter=Hyparam_set(self.args)
        self.args=self.hyperparameter.set_on()
     

        self.learner=Learner_config(self.args)
        self.args=self.learner.config()       

        self.model=self.learner.model
        self.optimizer=self.learner.optimizer
        self.optimizer_scheduler=self.learner.optimizer_scheduler

        self.dataloader=Dataloader_config(self.args)
        self.args=self.dataloader.config()

        mic_type=self.args['dataloader']['train']['mic_type']
        mic_num=self.args['dataloader']['train']['mic_num']
        mic=self.dataloader.train_loader.dataset.rir_maker.whole_mic_setup

        if mic_type=='whole':
            mic_pos=mic['mic_pos']
            mic_orV=mic['mic_orV']
            n_mic=mic_pos.shape[0]
            
        elif mic_type=='circular':
            if mic_num==4:
                mic_pos=mic['mic_pos'][:4]
                # mic_orV=mic['mic_orV'][:4]
            elif mic_num==6:
                mic_pos=mic['mic_pos'][4:10]
                # mic_orV=mic['mic_orV'][4:10]
            elif mic_num==8:
                mic_pos=mic['mic_pos'][10:18]

        elif mic_type=='ellipsoid':
            if mic_num==4:
                mic_pos=mic['mic_pos'][18:22]
                # mic_orV=mic['mic_orV'][:4]
            elif mic_num==6:
                mic_pos=mic['mic_pos'][22:28]
                # mic_orV=mic['mic_orV'][4:10]
            elif mic_num==8:
                mic_pos=mic['mic_pos'][28:36]

        elif mic_type=='linear':
            if mic_num==4:
                mic_pos=mic['mic_pos'][38:42]
                # mic_orV=mic['mic_orV'][:4]
            elif mic_num==6:
                mic_pos=mic['mic_pos'][37:43]
                # mic_orV=mic['mic_orV'][4:10]
            elif mic_num==8:
                mic_pos=mic['mic_pos'][36:]
        
        mic_array=mic_pos

    
        self.learner.beampattern_init(mic_array)
        

        self.logger=Logger_config(self.args)
        self.args=self.logger.config()
        

    
    def run(self, ):
      
   
        for epoch in range(self.args['hyparam']['resume_epoch'], self.args['hyparam']['last_epoch']):

          
            
            self.logger.epoch_init()
            
            
            self.train(epoch)
            self.test(epoch)           
            
            self.logger.epoch_finish(epoch, self.model, self.optimizer)
      

    def train(self, epoch):

        ######## train init
        self.model.train()
        
        self.optimizer.zero_grad()

        mic_type=self.args['dataloader']['train']['mic_type']
 

        for iter_num, (mixed, target_wav, azimuth, elevation, rotation_matrix) in enumerate(tqdm(self.dataloader.train_loader, desc='Train {}'.format(epoch), total=len(self.dataloader.train_loader), )):
        
            mixed=mixed.to(self.hyperparameter.device)
            target_wav=target_wav.to(self.hyperparameter.device)
       
            out, beamforming_weight_real, beamforming_weight_imag=self.model(mixed)
           
            beamforming_weight=torch.stack((beamforming_weight_real, beamforming_weight_imag), dim=-1)
            beamforming_weight=torch.view_as_complex(beamforming_weight)

            beampattern_power, target_tensor, target_direction_value=self.learner.get_pattern(beamforming_weight, azimuth, elevation, rotation_matrix)

          
        
            loss=self.learner.train_update(out, target_wav, beampattern_power, target_tensor)

            self.logger.train_iter_log(loss)
   
            self.learner.memory_delete([mixed, loss, out, beamforming_weight_imag, beamforming_weight_real, target_wav, beampattern_power, target_tensor, target_direction_value, beamforming_weight])
            gc.collect()
       
        self.logger.train_epoch_log()


    def test(self, epoch):
        self.model.eval()
        mic_type=self.args['dataloader']['test']['mic_type']
        with torch.no_grad():
            for iter_num, (mixed, target_wav) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader), )):
              
                mixed=mixed.to(self.hyperparameter.device)
                target_wav=target_wav.to(self.hyperparameter.device)
        

                out, beamforming_weight_real, beamforming_weight_imag=self.model(mixed)
                loss=self.learner.test_update(out, target_wav)

                self.logger.test_iter_log(loss)
      

                self.learner.memory_delete([mixed, loss, out, beamforming_weight_imag, beamforming_weight_real, target_wav])
    

            self.logger.test_epoch_log(self.optimizer_scheduler)
            




if __name__=='__main__':
    args=sys.argv[1:]
    
    args=util.util.get_yaml_args(args)
    t=Trainer(args)
    t.run()