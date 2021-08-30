import pandas as pd
import numpy as np
data_dir='/home/intern0/Desktop/project/IITP/Sound_source_localization/Data_processing/speech_setup/'


train_dir=data_dir+'test_audio.csv'

df=pd.read_csv(train_dir)
df=np.array(df['length'].tolist())
print(df.sum()/16000/3600)
