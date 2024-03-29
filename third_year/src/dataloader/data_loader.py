import os
from pathlib import Path
from torch.utils.data import DataLoader 
import pandas as pd
import soundfile as sf
import numpy as np
import random
import torch

from .random_gpu_rir_generator import gpu_rir_gen
import scipy
import cupyx
import cupyx.scipy.signal 
import cupy
import matplotlib.pyplot as plt
import pickle
from glob import glob
from natsort import natsorted
import gc

import math

import pickle

from scipy import signal


class Parameter:
	""" Random parammeter class.
	You can indicate a constant value or a random range in its constructor and then
	get a value acording to that with getValue(). It works with both scalars and vectors.
	"""
	def __init__(self, *args):
		if len(args) == 1:
			self.random = False
			self.value = np.array(args[0])
			self.min_value = None
			self.max_value = None
		elif len(args) == 2:
			self. random = True
			self.min_value = np.array(args[0])
			self.max_value = np.array(args[1])
			self.value = None
		else: 
			raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
	
	def getValue(self):
		if self.random:
			return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
		else:
			return self.value

class datamake():
    def __init__(self,):
        # self.gpuRIR_bind_simulator = gpuRIR_bind()
        self.eps = np.finfo(np.float32).eps
        
        self.fs=16000

        # snr mix
        # convolution
    
    def multi_ans(self, vad, azi_list, num_ans, degree_resolution):
        vad=torch.from_numpy(vad)

        
        
        if num_ans==0:
            return vad, torch.tensor(azi_list)

        vad=torch.repeat_interleave(vad, 2*num_ans+1, dim=0)
        
        new_azi=[]

        
        for azi in azi_list:
            temp_azi=torch.arange(azi-num_ans*degree_resolution, azi+(num_ans+1)*degree_resolution, degree_resolution)
            
            temp_azi=360+temp_azi
            temp_azi=torch.remainder(temp_azi, 360)
            new_azi.append(temp_azi)
        new_azi=torch.concat(new_azi)
        return vad, new_azi



    def gpu_convolve(self, audio, rir):
       
        
        audio_out=signal.fftconvolve(audio, rir, mode='full')   
      
        return audio_out

    def _cleanSilences(self, s, aggressiveness, return_vad=False):
        self.vad.set_mode(aggressiveness)

        vad_out = np.zeros_like(s)
        vad_frame_len = int(10e-3 * self.fs)
        n_vad_frames = len(s) // vad_frame_len
        for frame_idx in range(n_vad_frames):
            frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
            frame_bytes = (frame * 32767).astype('int16').tobytes()
            vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, self.fs)
        
        
        s_clean = s * vad_out
        
        return (s_clean, vad_out) if return_vad else s_clean



    def rms(self, data):
        """
        calc rms of wav
        """
        energy = data ** 2
        max_e = np.max(energy)
        low_thres = max_e*(10**(-50/10)) # to filter lower than 50dB 
        rms = np.mean(energy[energy>=low_thres])
        # rms=np.sqrt(rms)
        #rms = np.mean(energy)
        return rms

    def remove_dc(self, data):
        '''
        data: 1, T
        '''
        data_mean=data.mean()
        data=data-data_mean
        return data
     

    def snr_mix(self, clean, noise, snr):
        '''
        mix clean and noise according to snr
        '''
        

        clean_rms = self.rms(clean)
        clean_rms = np.maximum(clean_rms, self.eps)
        noise_rms = self.rms(noise)
        noise_rms = np.maximum(noise_rms, self.eps)
        k = math.sqrt(clean_rms / (10**(snr/10) * noise_rms))
        new_noise = noise * k
        return new_noise

    def scaling(self, data, normalize_factor):
        data=np.concatenate(data, axis=0)

        max_amp = np.max(np.abs(data))
        max_amp = np.maximum(max_amp, self.eps)
        scale=1. / max_amp * normalize_factor
        return scale

    def clipping(self, data, min=-1.0, max=1.0):

        return np.clip(data, min, max)

    def get_random_snr(self, white_noise_snr, normalize_factor):
        
        white_noise_snr=np.random.uniform(*white_noise_snr)
        normalize_factor=np.random.uniform(*normalize_factor)

        return white_noise_snr, normalize_factor
    def spk_mix(self, ):
        None
        
    def make_noisy(self,duration,  rired_speech_list,  white_noise_snr, normalize_factor, speech_start_point_list,  with_coherent_noise, coherent_noise_snr, rired_noise_wav):
        

        noisy_wav=np.zeros((rired_speech_list[0].shape[0], duration))

        for wav, start_point in zip(rired_speech_list, speech_start_point_list):
            noisy_wav[:,start_point:start_point+wav.shape[-1]]+=wav
        
        
        target_wav=noisy_wav[0:1]
        
        ##### white noise

        white_noise=np.random.normal(0, 1, noisy_wav.shape, ).astype('float32')
        white_noise=self.remove_dc(white_noise)
        noise=self.snr_mix(rired_speech_list[0], white_noise, white_noise_snr)

        if with_coherent_noise:
            
            rired_noise_wav=self.snr_mix(rired_speech_list[0], rired_noise_wav, coherent_noise_snr)

            noise=noise+rired_noise_wav

            noise=self.snr_mix(rired_speech_list[0], noise, coherent_noise_snr)
        
        noisy_wav=noisy_wav+noise
        scale=self.scaling([noisy_wav, target_wav], normalize_factor)

        noisy_wav=noisy_wav*scale
        target_wav=target_wav[0]*scale
        return noisy_wav, target_wav
        


     
    def get_vad(self, duration, vad_list, speech_start_point_list, max_spk):
        vad=np.zeros((max_spk, duration))
        for num, tmp in enumerate(vad_list):
            vad[num, speech_start_point_list[num]:speech_start_point_list[num]+tmp.shape[0]]=tmp
        return vad

    
    def add_noise(self, snr, white_noise_snr, normalize_factor, input_speech_wav, rired_noise_wav, ):

        
        ########## add noise

        
        
        rired_noise_wav=self.snr_mix(input_speech_wav, rired_noise_wav, snr)
        noisy=input_speech_wav+rired_noise_wav


        ####### normalize
        scale=self.scaling(noisy, normalize_factor)
 
        input_speech_wav=input_speech_wav * scale # rired clean
        early_speech_wav = early_speech_wav * scale # early rired clean
        noisy = noisy *  scale # noisy

        input_speech_wav=self.clipping(input_speech_wav)
        early_speech_wav=self.clipping(early_speech_wav)
        noisy=self.clipping(noisy)


        return input_speech_wav, early_speech_wav, noisy

    def fit_max_mic(self, data, max_mic):
     
        data=np.pad(data, ((0, max_mic-data.shape[0]), (0,0)))
        return data
        




    def snr_mixer(self,clean, noise,target_speech_wav, snr, normalize_factor=-25):
        EPS=1e-8
        # Normalizing to -25 dB FS
        rmsclean = (clean**2).mean()**0.5
        scalarclean = 10 ** (normalize_factor / 20) / (rmsclean+EPS)
        clean = clean * scalarclean
        rmsclean = (clean**2).mean()**0.5

        rmsnoise = (noise**2).mean()**0.5
        scalarnoise = 10 ** (normalize_factor / 20) /(rmsnoise+EPS)
        noise = noise * scalarnoise
        rmsnoise = (noise**2).mean()**0.5

        rmstarget=(target_speech_wav**2).mean()**0.5
        scalartarget = 10 ** (normalize_factor / 20) /(rmstarget+EPS)
        target_speech_wav = target_speech_wav * scalartarget
        
        # Set the noise level for a given SNR
        noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / (rmsnoise+EPS))
        noisenewlevel = noise * noisescalar
        noisyspeech = clean + noisenewlevel
        return noisyspeech, target_speech_wav, clean

    def snr_mixer_with_early(self, clean, noise, target_speech_wav, early_noise, snr, normalize_factor=-25):
        EPS=1e-8

        rmsclean = (target_speech_wav**2).mean()**0.5
        scalarclean = 10 ** (normalize_factor / 20) / (rmsclean+EPS)
        clean = clean * scalarclean


        rmsnoise = (early_noise**2).mean()**0.5
        scalarnoise = 10 ** (normalize_factor / 20) /(rmsnoise+EPS)
        noise = noise * scalarnoise
        rmsnoise = (noise**2).mean()**0.5
        scalartarget = 10 ** (normalize_factor / 20) /(rmsclean+EPS)
        target_speech_wav = target_speech_wav * scalartarget
   
        
        # Set the noise level for a given SNR
        noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / (rmsnoise+EPS))
        noisenewlevel = noise * noisescalar
        noisyspeech = clean + noisenewlevel
        return noisyspeech, target_speech_wav, clean

    def rir_peak_find(self,rir):
        rir_peak=np.argmax(np.abs(rir[0]))
        return rir_peak

    def early_rir(self, rir, early_reverb, peak):
        early_speech_rir=rir[0:1]
        early_speech_rir=early_speech_rir[:,:peak+early_reverb]
        return early_speech_rir

    def wav_save(self, data, dir, fs=16000):
        sf.write(dir, data.T, fs)



class train_data_loader(datamake):
    def __init__(self, args):    

        super(train_data_loader, self).__init__()   
        
        self.args=args
   

        self.noise_dir= self.args['noise_dir']
        self.speech_dir= self.args['speech_dir']

        self.metadata_dir=self.args['metadata_dir']
        self.ans_azi=self.args['ans_azi']
        self.degree_resolution=self.args['degree_resolution']        
        
        self.noise_csv=pd.read_csv(self.metadata_dir+self.args['noise_csv'], index_col=0)       
        self.speech_csv=pd.read_csv(self.metadata_dir+self.args['speech_csv'], index_col=0)        
    
        
        ########### hyperparameter
        
        self.duration=self.args['duration']        
        self.fs=self.args['fs']
        self.early_reverb=self.args['early_reverb']*self.fs//1000
        self.max_n_mic=self.args['max_n_mic']
        self.least_chunk_size=self.args['speech_least_chunk_size']
        

        ######### random
        self.snr=self.args['SNR']
        self.max_num_people=self.args['max_spk']
        self.normalize_factor_bound=self.args['normalize_factor']        
        self.white_noise_snr=self.args['white_noise_snr']        
        self.without_coherent_noise_probability=self.args['without_coherent_noise']

        ############ rir
        
        self.rir_maker=gpu_rir_gen.acoustic_simulator_on_the_fly(self.args)    

        
            
        
    def __len__(self):

        self.DB_size_per_epoch=self.args['iteration_num_per_epoch']*self.args['dataloader_dict']['batch_size']

        if self.DB_size_per_epoch>len(self.speech_csv):
            self.DB_size_per_epoch=len(self.speech_csv)

   
        return self.DB_size_per_epoch
    def get_speech_start_point(self, speech_wav, rir_peak, pos, vad, ):
        
        vad_len=vad.shape[-1]
        rired_len=speech_wav.shape[-1]
        if pos=='full':
            start_point=0
            speech_wav=speech_wav[:, rir_peak:rir_peak+self.duration]

        elif pos=='mid':
            
            if rired_len>self.duration:
                speech_wav=speech_wav[:, rir_peak:self.duration+rir_peak]
            rired_len=speech_wav.shape[-1]
            start_point=random.randint(0, self.duration-rired_len)

        
            vad=np.pad(vad, ( 0, rired_len-vad_len))


            
        elif pos=='front':
            back_cut=rired_len-vad_len-rir_peak
            speech_wav=speech_wav[:, :-back_cut]
            rired_len=speech_wav.shape[-1]

            if rired_len>self.duration:
                start_point=0
                speech_wav=speech_wav[:, -self.duration]
                vad=np.pad(vad, ( self.duration-vad_len, 0))

            else:
                start_point=self.duration-rired_len
                vad=np.pad(vad, ( rir_peak, 0))


        elif pos=='back':
            front_cut=rir_peak
            speech_wav=speech_wav[:, front_cut:]
            rired_len=speech_wav.shape[-1]

            if rired_len>self.duration:
                start_point=0
                speech_wav=speech_wav[:, :self.duration]
                vad=np.pad(vad, (0, self.duration-vad_len))
            else:
                start_point=0
                vad=np.pad(vad, ( 0, rired_len-vad_len))



            
        
            
        return start_point, speech_wav, vad


    def speech_get_wav(self, wav_file, vad):

        speech_wav, _ = sf.read(wav_file, dtype='float32')
        speech_total_duration=speech_wav.shape[0]

        vad=np.ones_like(speech_wav)
        

        ######## select postion
        if speech_total_duration<self.least_chunk_size:
            pos='mid'
        elif speech_total_duration>self.duration:
            pos=random.choice(['full', 'back', 'front'])
        else:
            pos=random.choice(['front', 'back', 'mid'])

        ###### get chunk
        speech_start_sample=0
        if pos=='full':
            speech_start_sample=random.randrange(0, speech_total_duration-self.duration)        
            speech_wav=speech_wav[speech_start_sample:speech_start_sample+self.duration,]
            vad=vad[speech_start_sample:speech_start_sample+self.duration,]
            start_point=0


        elif pos=='front': 
            new_duration=random.randrange(self.least_chunk_size, self.duration)
    
            speech_wav=speech_wav[speech_start_sample:speech_start_sample+new_duration,]
            vad=vad[speech_start_sample:speech_start_sample+new_duration,]
            start_point=0

        elif pos=='back':
            new_duration=random.randrange(self.least_chunk_size, self.duration)
            speech_wav=speech_wav[-new_duration:,]
            vad=vad[-new_duration:,]
            start_point=self.duration-new_duration

    
        elif pos=='mid':

            
            start_point=random.randint(0, self.duration-speech_total_duration)

        speech_wav=self.remove_dc(speech_wav)
        return speech_wav, pos, start_point, vad

    def select_different_speakers(self, speech_info, num_spk):
        temp_total_df=self.speech_csv
     

        for spk in range(num_spk-1): # selecting num_spk-1 more speakers, not overlapped
            last_speaker=speech_info.iloc[-1]['speaker_id']
            temp_total_df=temp_total_df.drop(temp_total_df[temp_total_df['speaker_id']==last_speaker].index)
            temp_df=temp_total_df.sample(1)            
            speech_info=pd.concat((speech_info, temp_df))
        
        return speech_info

    def main_speech_load(self, speech_info):
        
        wav_file=self.speech_dir+speech_info['file_path']

        vad=0
        

        return self.speech_get_wav(wav_file, vad)


    def noise_load(self, rir_peak):
        noise_info=self.noise_csv.sample(n=1)

        noise_total_duration=noise_info['length'].iloc[0]

        noise_start_sample=0
        if noise_total_duration>self.duration:
            noise_start_sample=random.randrange(0, noise_total_duration-self.duration)
            padding_size=0            
            get_duration=self.duration
        
        else:
            padding_size=self.duration-noise_total_duration
            get_duration=-1
        
        
        noise_wav, _ = sf.read(self.noise_dir+noise_info['noise_directory'].iloc[0], dtype='float32', start=noise_start_sample, frames=get_duration)
        
        if padding_size>0:
            front_padding=np.random.randint(0, padding_size)
            noise_wav=np.pad(noise_wav, (front_padding, padding_size-front_padding))
        
        
        

        noise_wav=np.expand_dims(noise_wav, 0)

        noise_wav=self.remove_dc(noise_wav)
       
        return noise_wav

    def get_full_wav(self, speech_info):
        wav_file=self.speech_dir+speech_info['file_path']
        y, sr=sf.read(wav_file ,dtype='float32')

        return y
    def spk_mixer(self, rired_speech_list):
        ref_wav=rired_speech_list[0]
        spk_snr_list=[0.0]

        for spk_num in range(len(rired_speech_list[0:])):
            spk_snr=random.uniform(*self.args['spk_SNR'])
            spk_snr_list.append(spk_snr)
            other_spk=self.snr_mix(ref_wav, rired_speech_list[spk_num], spk_snr)
            rired_speech_list[spk_num]=other_spk
   


        return rired_speech_list, spk_snr_list


    def  __getitem__(self, idx):            

          
                    
                    
                    num_spk=random.randint(1, self.max_num_people)  
               
                    

                    ###### rir
                    with_coherent_noise=random.choices([True, False], weights=self.without_coherent_noise_probability, k=1)[0]  
               
                
                    rirs, mic_pos, azi_list, linear_azi_pos_list, ele_list = self.rir_maker.create_rir(num_spk=num_spk, with_coherent_noise=with_coherent_noise, mic_type=self.args['mic_type'], mic_num=self.args['mic_num'])

                    self.mic_pos=mic_pos
             
                    
                    coherent_noise_snr=None
                    rired_noise_wav=None
                    ####### coherent noise
                    if with_coherent_noise:
                        noise_rir=rirs[-1]
                        self.noise_rir_peak=self.rir_peak_find(noise_rir)
                        noise_wav=self.noise_load(self.noise_rir_peak)
                        
                        rired_noise_wav=self.gpu_convolve(noise_wav, noise_rir)[:,self.noise_rir_peak:self.duration+self.noise_rir_peak]
                        
                        coherent_noise_snr=np.random.uniform(*self.args['SNR'])
                    

                    speech_rirs=rirs[:num_spk]
                    azi_list=azi_list[:num_spk]
                                     

             
                    ##### speech
                    
                    rired_speech_list=[]
        
                    speech_start_point_list=[]
                    speech_info=self.select_different_speakers(self.speech_csv.iloc[idx:idx+1], num_spk)
     

                    for spk_num, spk_info in enumerate(speech_info.iterrows()):
                        spk_info=spk_info[1]          

                        speech_wav, pos, speech_start_point, vad_out=self.main_speech_load(spk_info)

                        
                        s_clean=speech_wav

                      
                        
                        speech_rir=speech_rirs[spk_num]       
                        self.speech_rir_peak=self.rir_peak_find(speech_rir)
                        s_clean=np.expand_dims(s_clean, 0)
                        rired_speech=self.gpu_convolve(s_clean, speech_rir)
                        
                        
                        start_point, rired_speech, vad_out=self.get_speech_start_point(rired_speech, self.speech_rir_peak, pos, vad_out, )

                        
                        rired_speech_list.append(rired_speech)

                        # vad_out=vad
                        
                        # vad_list.append(vad_out)
                        speech_start_point_list.append(start_point)
                    
                    
                    white_noise_snr, normalize_factor=self.get_random_snr(self.white_noise_snr, self.normalize_factor_bound)


                    ####### spk mixer, normalizing speech

                    
                    rired_speech_list, spk_snr_list=self.spk_mixer(rired_speech_list)
                 
                    

                    ######## speech & noise   

                    mixed, target_wav=self.make_noisy(self.duration, rired_speech_list, white_noise_snr, normalize_factor, speech_start_point_list, with_coherent_noise, coherent_noise_snr, rired_noise_wav)
                    
                    mixed=self.clipping(mixed)

                    for i in range(self.max_num_people-num_spk):
                        azi_list.append(0)
                    mixed=mixed.astype('float32')
                    target_wav=target_wav.astype('float32')
                    mixed=self.fit_max_mic(mixed, self.args['max_n_mic'])
           
                    
                    rotation_matrix=self.get_rotation_matrix(azi_list[0], ele_list[0])

                    gc.collect()
                
               
                    return torch.from_numpy(mixed), torch.from_numpy(target_wav), np.deg2rad(azi_list[0], dtype=np.float32), np.deg2rad(ele_list[0], dtype=np.float32), rotation_matrix

    def sph2cart(self, az, el, r=1.0):
        x=r*np.sin(el)*np.cos(az)
        y=r*np.sin(el)*np.sin(az)
        z=r*np.cos(el)

    

        return x, y, z               


    def get_rotation_matrix(self, azi, ele, ref_vector=np.array([0, 0, 1])):

        
        azi=np.deg2rad(azi)
        ele=np.deg2rad(ele)
        x, y, z=self.sph2cart(azi, ele)
        target_vector=np.array([x, y, z])
    
        
        
        a, b = (ref_vector / np.linalg.norm(ref_vector)).reshape(3), (target_vector / np.linalg.norm(target_vector)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v) #+ 1e-8
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        if np.isnan(rotation_matrix).any():
        
       
            rotation_matrix=-np.eye(3)

        rotation_matrix=rotation_matrix.astype(np.float32)


        return rotation_matrix
    

class validate_data_loader(datamake):
    def __init__(self, config):

        super(validate_data_loader, self).__init__()

        self.noise_dir= config['noise_dir']
        self.speech_dir=config['speech_dir']
        
        data_csv_dir=config['data_csv_dir']
        self.datacsv=data_csv_dir+config['data_csv']
        self.datacsv=pd.read_csv(self.datacsv, index_col=0)
        self.data_dict_dir=config['data_dict_dir']
        
        self.duration=config['duration']
        self.fs=config['fs']
        self.max_n_mic=config['max_n_mic']
        self.early_reverb=config['early_reverb']*self.fs//1000
       
        self.rir_maker=gpu_rir_gen.acoustic_simulator_for_test()



    def __len__(self):
        return len(self.datacsv)

    def get_wav(self, wav_name, start_point):
        wav, _ = sf.read(wav_name,dtype='float32', start=start_point, stop=start_point+self.duration, fill_value=0.0)

        wav=np.pad(wav, (0, self.duration-wav.shape[0]))

        wav=np.expand_dims(wav, 0)
        wav=self.remove_dc(wav)
        return wav

    def  __getitem__(self, idx):
        data_row=self.datacsv.loc[idx]
        data_dir=self.data_dict_dir+data_row['data_list']
        pkl_file = open(data_dir, 'rb')
        data_dict = pickle.load(pkl_file)

        pkl_file.close()

        ###### rir
        speech_rir, noise_rir, mic_pos=self.rir_maker.create_rir(data_dict['gpu_rir_generate_dict'][0])
        

        self.speech_rir_peak=self.rir_peak_find(speech_rir)
        self.speech_early_rir=self.early_rir(speech_rir, self.early_reverb, self.speech_rir_peak)


        self.noise_rir_peak=self.rir_peak_find(noise_rir)

        ######## speech & noise  
        noise_wav=self.get_wav(self.noise_dir+data_dict['noise_wav'], data_dict['noise_start_point'])
        speech_wav=self.get_wav(self.speech_dir+data_dict['speech_wav'], data_dict['speech_start_point'])

        ######### convolution, add reverb
        input_speech_wav=self.gpu_convolve(speech_wav, speech_rir)[:, :self.duration]
        early_speech_wav=self.gpu_convolve(speech_wav, self.speech_early_rir)[:, :self.duration]

        rired_noise_wav=self.gpu_convolve(noise_wav, noise_rir)[:,self.noise_rir_peak:self.duration+self.noise_rir_peak]


        ########### noise add
        rired_clean_wav, early_clean_wav, noisy=self.add_noise(data_dict['SNR'], None, data_dict['normalize_factor'], input_speech_wav, early_speech_wav, rired_noise_wav, add_white_noise=False)


        
        noisy=self.fit_max_mic(noisy, self.max_n_mic)
        rired_clean_wav=self.fit_max_mic(rired_clean_wav, self.max_n_mic)

        mic_pos=mic_pos-mic_pos[0]

        mic_pos=self.fit_max_mic(mic_pos, self.max_n_mic).astype('float32')

        return noisy, rired_clean_wav, early_clean_wav, mic_pos


class IITP_test_loader(datamake):
    def __init__(self, args):

        super(IITP_test_loader, self).__init__()
        self.args=args
        self.room_type=None

        self.metadata_dir=self.args['metadata_dir']
        self.pkl_dir=self.args['pkl_dir']

        self.ans_azi=self.args['ans_azi']
        self.degree_resolution=self.args['degree_resolution']  
        
        

        pkl_list=os.listdir(self.pkl_dir+'409')
        self.pkl_list=natsorted(pkl_list)
        

        self.noisy_pkl_csv=pd.read_csv(self.metadata_dir+self.args['pkl_csv'], index_col=0) 
 
    def __len__(self):
        return len(self.pkl_list)

    def  __getitem__(self, idx):
        
   

        data_row=self.pkl_list[idx]
        

        
 

        data_dir=self.pkl_dir+str(self.room_type)+'/'+data_row
        
        

        pkl_file = open(data_dir, 'rb')
        data_dict = pickle.load(pkl_file)
        pkl_file.close()

       
        mic_type=self.args['mic_type']
        mic_num=self.args['mic_num']

        mixed=data_dict['noisy']
        target_wav=data_dict['target_wav']
        target_wav=target_wav.type(torch.float32)

        mixed=mixed[:self.args['max_n_mic']]
        
       
        
        return mixed, target_wav, data_row



class testset_maker_loader(datamake):
    def __init__(self, args):

        super(testset_maker_loader, self).__init__() 
        
        self.args=args
        self.noise_dir= self.args['noise_dir']
        self.speech_dir= self.args['speech_dir']
        self.vad_dir=self.args['vad_dir']
        self.metadata_dir=self.args['metadata_dir']
        self.pkl_dir=self.args['pkl_dir']

        self.ans_azi=self.args['ans_azi']
        self.degree_resolution=self.args['degree_resolution']  
        
        
 
        self.noise_csv=pd.read_csv(self.metadata_dir+self.args['noise_csv'], index_col=0)
        self.speech_csv=pd.read_csv(self.metadata_dir+self.args['speech_csv'], index_col=0) 


      
        

        ########### hyperparameter
        
        self.duration=self.args['duration']        
        self.fs=self.args['fs']
        self.early_reverb=self.args['early_reverb']*self.fs//1000
        self.max_n_mic=self.args['max_n_mic']
        self.least_chunk_size=self.args['speech_least_chunk_size']

        ######### random
        self.snr=self.args['SNR']
        self.max_num_people=self.args['max_spk']
        self.normalize_factor_bound=self.args['normalize_factor']        
        self.white_noise_snr=self.args['white_noise_snr']        
        

        ############ rir
        
        self.rir_maker=gpu_rir_gen.acoustic_simulator_on_the_fly(self.args) 

        
    def noise_load(self, rir_peak):
        
        noise_dir=random.choice(self.noise_csv)
        noise_wav, fs = sf.read(noise_dir, dtype='float32', )
        noise_wav=scipy.signal.decimate(noise_wav, int(fs/self.fs), axis=0)
        fs=self.fs
    
        
        
        noise_total_duration=noise_wav.shape[0]

        noise_start_sample=0
        if noise_total_duration>self.duration:
            noise_start_sample=random.randrange(0, noise_total_duration-self.duration)
            padding_size=0            
            get_duration=self.duration
            noise_wav=noise_wav[noise_start_sample:noise_start_sample+get_duration]
        
        else:
            padding_size=self.duration-noise_total_duration
            get_duration=-1
        
        
        
        
        if padding_size>0:
            front_padding=np.random.randint(0, padding_size)
            noise_wav=np.pad(noise_wav, (front_padding, padding_size-front_padding))
        
        
        

        noise_wav=np.expand_dims(noise_wav, 0)

        noise_wav=self.remove_dc(noise_wav)
       
        return noise_wav

    def select_different_speakers(self, speech_info, num_spk):
        temp_total_df=self.speech_csv
        
     

        for spk in range(num_spk-1): # selecting num_spk-1 more speakers, not overlapped
            last_speaker=speech_info.iloc[-1]['speaker']
            temp_total_df=temp_total_df.drop(temp_total_df[temp_total_df['speaker']==last_speaker].index)
            temp_df=temp_total_df.sample(1)            
            speech_info=pd.concat((speech_info, temp_df))
      
        return speech_info

    def speech_get_wav(self, wav_file, vad):

        speech_wav, _ = sf.read(wav_file, dtype='float32')
        speech_total_duration=speech_wav.shape[0]
        

        ######## select postion
        if speech_total_duration<self.least_chunk_size:
            pos='mid'
        elif speech_total_duration>self.duration:
            pos=random.choice(['full', 'back', 'front'])
        else:
            pos=random.choice(['front', 'back', 'mid'])

        ###### get chunk
        speech_start_sample=0
        if pos=='full':
            speech_start_sample=random.randrange(0, speech_total_duration-self.duration)        
            speech_wav=speech_wav[speech_start_sample:speech_start_sample+self.duration,]
            vad=vad[speech_start_sample:speech_start_sample+self.duration,]
            start_point=0


        elif pos=='front': 
            new_duration=random.randrange(self.least_chunk_size, self.duration+1)
            
            speech_wav=speech_wav[speech_start_sample:speech_start_sample+new_duration,]
            vad=vad[speech_start_sample:speech_start_sample+new_duration,]
            start_point=0

        elif pos=='back':
            new_duration=random.randrange(self.least_chunk_size, self.duration+1)

            speech_wav=speech_wav[-new_duration:,]
            vad=vad[-new_duration:,]
            start_point=self.duration-new_duration

           
        elif pos=='mid':

            

            
            start_point=random.randint(0, self.duration-speech_total_duration)

        speech_wav=self.remove_dc(speech_wav)
        return speech_wav, pos, start_point, vad

    def get_speech_start_point(self, speech_wav, rir_peak, pos, vad, ):
        
        vad_len=vad.shape[-1]
        rired_len=speech_wav.shape[-1]
        if pos=='full':
            start_point=0
            speech_wav=speech_wav[:, rir_peak:rir_peak+self.duration]

        elif pos=='mid':
            
            if rired_len>self.duration:
                speech_wav=speech_wav[:, rir_peak:self.duration+rir_peak]
            rired_len=speech_wav.shape[-1]
            start_point=random.randint(0, self.duration-rired_len)

        
            vad=np.pad(vad, ( 0, rired_len-vad_len))


            
        elif pos=='front':
            back_cut=rired_len-vad_len-rir_peak
            speech_wav=speech_wav[:, :-back_cut]
            rired_len=speech_wav.shape[-1]

            if rired_len>self.duration:
                start_point=0
                speech_wav=speech_wav[:, -self.duration]
                vad=np.pad(vad, ( self.duration-vad_len, 0))

            else:
                start_point=self.duration-rired_len
                vad=np.pad(vad, ( rir_peak, 0))


        elif pos=='back':
            front_cut=rir_peak
            speech_wav=speech_wav[:, front_cut:]
            rired_len=speech_wav.shape[-1]

            if rired_len>self.duration:
                start_point=0
                speech_wav=speech_wav[:, :self.duration]
                vad=np.pad(vad, (0, self.duration-vad_len))
            else:
                start_point=0
                vad=np.pad(vad, ( 0, rired_len-vad_len))



            
        
            
        return start_point, speech_wav, vad
    def spk_mixer(self, rired_speech_list):
        ref_wav=rired_speech_list[0]
        spk_snr_list=[0.0]

        for spk_num in range(len(rired_speech_list[0:])):
            spk_snr=random.uniform(*self.args['spk_SNR'])
            spk_snr_list.append(spk_snr)
            other_spk=self.snr_mix(ref_wav, rired_speech_list[spk_num], spk_snr)
            rired_speech_list[spk_num]=other_spk
   

        return rired_speech_list, spk_snr_list
    def main_speech_load(self, speech_info):
        
        wav_file=self.speech_dir+speech_info['audio_directory']

        vad_name=speech_info['audio_directory'][6:]
        vad_name=vad_name[:6]+'_'+vad_name[7:]
        
       
        vad=self.vad_dir+vad_name.replace('.wav', '.npy')
        vad=np.load(vad)
        

        return self.speech_get_wav(wav_file, vad)  

    def __len__(self):
        return len(self.speech_csv)

    def make_test_set(self, idx):
        num_spk=random.randint(1, self.max_num_people) 
   
        with_coherent_noise=True
        
        
        rirs, mic_pos, azi_list = self.rir_maker.create_rir(num_spk=num_spk, with_coherent_noise=with_coherent_noise)
        

        coherent_noise_snr=None
        rired_noise_wav=None
        ####### coherent noise
        if with_coherent_noise:
            noise_rir=rirs[-1]
            self.noise_rir_peak=self.rir_peak_find(noise_rir)
            noise_wav=self.noise_load(self.noise_rir_peak)
            
            rired_noise_wav=self.gpu_convolve(noise_wav, noise_rir)[:,self.noise_rir_peak:self.duration+self.noise_rir_peak]
            
            coherent_noise_snr=np.random.uniform(*self.args['SNR'])
        speech_rirs=rirs[:num_spk]
        azi_list=azi_list[:num_spk]

   
                              

    
        ##### speech
        
        rired_speech_list=[]
      
        vad_list=[]
        speech_start_point_list=[]
        speech_info=self.select_different_speakers(self.speech_csv.iloc[idx:idx+1], num_spk)
 

        for spk_num, spk_info in enumerate(speech_info.iterrows()):
            spk_info=spk_info[1]          

            speech_wav, pos, speech_start_point, vad_out=self.main_speech_load(spk_info)

            
            s_clean=speech_wav

            
            
            speech_rir=speech_rirs[spk_num]       
            self.speech_rir_peak=self.rir_peak_find(speech_rir)
            s_clean=np.expand_dims(s_clean, 0)
            rired_speech=self.gpu_convolve(s_clean, speech_rir)
            
            
            start_point, rired_speech, vad_out=self.get_speech_start_point(rired_speech, self.speech_rir_peak, pos, vad_out, )

            
            rired_speech_list.append(rired_speech)
        
            vad_list.append(vad_out)
            speech_start_point_list.append(start_point)
        
        
        white_noise_snr, normalize_factor=self.get_random_snr(self.white_noise_snr, self.normalize_factor_bound)


        ####### spk mixer, normalizing speech

        
        rired_speech_list, spk_snr_list=self.spk_mixer(rired_speech_list)
        ########### get vad
        vad=self.get_vad(self.duration, vad_list, speech_start_point_list, self.max_num_people)
        

        ######## speech & noise   

        mixed=self.make_noisy(self.duration, rired_speech_list, white_noise_snr, normalize_factor, speech_start_point_list, with_coherent_noise, coherent_noise_snr, rired_noise_wav)
        
        mixed=self.clipping(mixed)

        for i in range(self.max_num_people-num_spk):
            azi_list.append(0)
        mixed=mixed.astype('float32')
        vad=vad.astype('float32')


        save_dict={}
        save_dict['noisy']=torch.from_numpy(mixed)
        save_dict['vad']=torch.from_numpy(vad)
        save_dict['azi']=torch.tensor(azi_list)
        save_dict['num_spk']=num_spk
        save_dict['SNR']=coherent_noise_snr

    
        
        pkl_name=self.pkl_dir+str(idx)+'.pkl'

        output=open(pkl_name, 'wb')
        pickle.dump(save_dict, output)
        output.close()
      

        return 1,2,3,4
        
    def  __getitem__(self, idx):

        return self.make_test_set(idx)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed) 



def IITP_test_dataload(config):
    return DataLoader(IITP_test_loader(config),
                                             pin_memory=True,
                                             **config['dataloader_dict']
                                             )




def Test_data_maker_load(config):
    return DataLoader(testset_maker_loader(config),
                                             pin_memory=True,
                                             **config['dataloader_dict']
                                             )

def Train_dataload(config, init_seed):
    g = torch.Generator()    
    g.manual_seed(init_seed)

    return DataLoader(train_data_loader(config),
                                            pin_memory=True,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            **config['dataloader_dict']
                                            )

def Validate_dataload(config, device, init_seed):
    g = torch.Generator()    
    g.manual_seed(init_seed)
  
     
    return DataLoader(validate_data_loader(config),
                                            pin_memory=True,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            **config['dataloader_dict']
                                            )