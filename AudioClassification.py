# Imports
#%%
import os
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio
import pandas as pd
import librosa as lr
import numpy as np
from sklearn.model_selection import train_test_split
#%%
# Define paths
data_folder='data'
print(pd.DataFrame(os.listdir(data_folder),columns=['Files']))

def count(path):
    size = []
    for file in os.listdir(path):
        size.append(len(os.listdir(os.path.join(path,file))))
    return pd.DataFrame(size,columns=['Number Of Sample'],index=os.listdir(path))  
    
sample_counts = count(data_folder)
print(sample_counts)

def load(path):
    data=[]
    label=[]
    sample=[]
    for file in os.listdir(path):
        path_=os.path.join(path,file)
        for fil in os.listdir(path_):
            data_contain,sample_rate=lr.load(os.path.join(path_,fil) ,sr=16000)
            data.append(data_contain)
            sample.append(sample_rate)
            label.append(file)
    return data,label,sample

data,label,sample=load(data_folder)
df=pd.DataFrame()
df['Label'],df['sample']=label,sample
print(df)
#%%
# Waveform

def waveform(data,sr,label):
    plt.figure(figsize=(14, 5))
    lr.display.waveshow(data, sr=sr)
    plt.suptitle(label)
    plt.title('Waveform plot')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

# MFCC features

def mfcc(data, sr):
    mfccs = lr.feature.mfcc(y=data, sr=sr)
    return np.mean(mfccs), mfccs

def mfcc_v(mfccs,label):
    plt.figure(figsize=(10, 4))
    lr.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.suptitle(label)

# Mel-spectrogram

def Mel(data, sr):
    mel_spec = lr.feature.melspectrogram(y=data, sr=sr)
    return np.mean(mel_spec), mel_spec

def mel_v(mel_spec,label,sr):
    # Convert to decibel scale
    mel_spec_db = lr.power_to_db(mel_spec, ref=np.max)
    # Visualize Mel-spectrogram
    plt.figure(figsize=(10, 4))
    lr.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.suptitle(label)

# zero_crossing_rate

def zero_crossing(data,sr):
    # Compute zero-crossing rate
    zcr = lr.feature.zero_crossing_rate(data)
    # Print average zero-crossing rate
    avg_zcr = sum(zcr[0])/len(zcr[0])
    print("Average zero-crossing rate:", avg_zcr)
    return zcr

def zero_crossing_v(zcr,label,data,sr):
    time = lr.times_like(zcr)
    # Create waveform plot
    plt.figure(figsize=(14, 5))
    lr.display.waveshow(data, sr=sr, alpha=0.5)
    plt.plot(time, zcr[0], color='r')
    plt.title('Zero-crossing rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.suptitle(label)

# Show test Waveform
waveform(data[0],sample[0],label[0])
plt.legend()
plt.show()
#%%

# Show test MFCC

mfccs_mean, mfccs = mfcc(data[0], sample[0])
print('MFCCs Mean:', mfccs_mean)
print('MFCCs shape:', mfccs.shape)
mfcc_v(mfccs,label[0])
plt.show()
# %%
# Show test Spectrogram

mel_mean,mel=Mel(data[0],sample[0])
print('Mel Mean:',mel_mean)
print('Mel :',mel.shape)
mel_v(mel,label[0],sample[0])
# %%

# Create numbered labels
code = {}
x = 0
for i in pd.unique(label):
    code[i] = x
    x += 1
print(pd.DataFrame(code.values(),columns=['Value'],index=code.keys()))

# Label data with new labels

def get_Name(N):
    for x,y in code.items():
          if y==N:
                return x
for i in range(len(label)):
    label[i]=code[label[i]]
print(pd.DataFrame(label,columns=['Labels']))   

# %%

# Build data train/test

def pad_or_truncate(array, target_length=16000):
    if len(array) < target_length:
        return np.pad(array, (0, target_length - len(array)), mode='constant')
    return array[:target_length]
data = [pad_or_truncate(d, target_length=16000) for d in data]

data=np.array(data).reshape(-1,16000,1)
label=np.array(label)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=44, shuffle =True)
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)

# %%
 