from torch.utils.data import DataLoader 
import pandas as pd
import soundfile as sf
import numpy as np

class wav_data_loader():
    def __init__(self, config):
        csv=pd.read_csv(config['csv'])
        self.input_list=csv['input_path'].tolist()
        self.label_list=csv['label_path'].tolist()
        self.duration=config['duration']
        self.location=config['audio_path']
        self.snr=csv['SNR'].tolist()
        
        
    def __len__(self):
        return len(self.input_list)

    def  __getitem__(self, idx):
        input_file=self.input_list[idx]
        input_file=self.location+self.input_list[idx][9:]
        input_file, _ = sf.read(input_file, dtype='float32')

        label_file=self.label_list[idx]
        label_file=self.location+self.label_list[idx][9:]
        label_file,_=sf.read(label_file, dtype='float32')

        if self.duration == 'None':
            return input_file.T, label_file.T, self.snr[idx], self.input_list[idx]

        if input_file.shape[0]>self.duration:
            start=np.random.randint(0, input_file.shape[0]-self.duration)
            input_file=input_file[start:start+self.duration,:]
            label_file=label_file[start:start+self.duration]

        elif input_file.shape[0]<self.duration:
            gap=self.duration-input_file.shape[0]
            front=np.random.randint(0, gap)
            back=gap-front
            input_file=np.pad(input_file, ((front, back), (0,0)))
            label_file=np.pad(label_file, (front, back))

        

        return input_file.T, label_file.T



def  Custom_dataload(config):
     
    return DataLoader(wav_data_loader(config),
                                             batch_size=config['batch_size'],
                                             shuffle=config['shuffle'],
                                             num_workers=config['num_workers'],
                                             pin_memory=True,
                                             drop_last=config['drop_last'])
