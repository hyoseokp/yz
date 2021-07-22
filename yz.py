#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, transforms
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import soundfile as sf
import os
from tqdm import tqdm
# sig_s, sr_s = librosa.load('sound1.wav' , sr = 16000)
# sig_m, sr_m = librosa.load("Coldplay_Don't Panic.wav", sr = 16000)


# In[9]:


sig, sr = librosa.load('yz/Benjamin Francis Leftwich-04-Is That You On That Plane-320k.wav',sr=16000)
folder = 'yz'
for root, dirs, files in os.walk(folder):
#     sig, sr = librosa.load('yz/%s'%files,sr=16000)
    for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(data_type)):
# for i in range(0,len())
print(os.walk(folder))


# In[34]:


folder = 'yz'
window_size = 16000
stride = 1/2
sample_rate = 16000
serialized_folder = 'yz_save'
def slice_signal(file, window_size, stride, sample_rate):
 
    wav, sr = librosa.load(file, sr=sample_rate)
    # wav, sr = sf.read(file)
    if sr != 16000: print('asdasd');
    hop = int(window_size *stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


for root, dirs, files in os.walk(folder):

    for filename in files:
        x = os.path.join(folder, filename)
        s = slice_signal(x, window_size, stride, sample_rate)

        for idx, slice_tuple in enumerate(s):
            save_file = np.array(slice_tuple)
            print(idx,slice_tuple)
            np.save(os.path.join(serialized_folder, '{}_{}'.format(filename, idx)), arr=save_file)


# In[30]:


A=np.array([[1,2,3,4,5],[1,3,5,7,9],[2,4,6,8,0]])
print(A)
print(zip(A))
for a in enumerate(zip(A)):
    print(a)
for a in enumerate(A):
    print(a)


# In[ ]:





