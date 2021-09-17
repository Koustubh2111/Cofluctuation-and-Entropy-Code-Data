#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import json
import seaborn as sns
import sys
from statannot import add_stat_annotation

def getEntropy(filename):
    #Input : Filename - The path of the .csv file containing entropy series
    #Output : Extracted entropy
    df = pd.read_csv(filename)
    time = df['time']
    entropy = df['shannon_entropy']
    
    #-999 values are removed
    index = entropy > 0
    time = time[index]
    entropy = entropy[index]
    
    #Remove repeated values
    filtered_entropy_index = np.nonzero(np.abs(np.diff(entropy)))[0]
    
    #Convert to entropy to array to prevent old version depreciation
    entropy = np.array(entropy)
    time = np.array(time)
    
    entropy = entropy[filtered_entropy_index]
    time = time[filtered_entropy_index]    
    
    return time, entropy

def getMeanStdEntropy(entropy):    
    #Input : entropy - extracted entropy from getEntropy
    #Output : Mean and Std of entropy
    return np.mean(entropy), np.std(entropy)

# In[ ]:
root_dir = '../NormalAnimals'
animal_names = os.listdir(root_dir) #Normals

normal_mean_entropy = []
normal_std_entropy = []

normal_animal_mean = []
normal_animal_std = []

channel_entropy_baseline = []
# entropy_filename = 'normals_entropy_mean.csv'
entropy_filename = 'normals_entropy_std.csv'


for animal in animal_names:
    
    animal_attmetric_dir = root_dir + '/' + animal + '/AttentionMetric_1min_20minbuff/'
    channel_mean = []
    channel_std = []
    
    #Looping through 16 channels in animal 
    for channel in sorted(os.listdir(animal_attmetric_dir)):
        
        channel_dir = animal_attmetric_dir + channel + '/'
        
        channel_csv_list = [file for file in glob.glob( channel_dir +"*.csv")]
        
        #Get all the event names including baseline
        event_names = [event_filename.split('\\', 10)[-1] for event_filename in channel_csv_list]

        #isoalte baseline csv
        event_name = [event_name for event_name in event_names \
            if ("0_shannon" in event_name) and (event_name[event_name.find("0_shannon") - 1] != '1')][0]
            
        #print(animal, channel, event_name)
        event_filename = channel_dir + event_name
        
        time, entropy = getEntropy(event_filename)
        
        channel_entropy_baseline.append({'animal': animal, 'event_name': event_name, 'channel': channel, 'entropy': np.std(np.array(entropy))})
        
        d = pd.DataFrame(channel_entropy_baseline)
        d.to_csv(entropy_filename, index=False)
        
        channel_mean_baseline, channel_std_baseline = getMeanStdEntropy(entropy)
        
        channel_mean.append(channel_mean_baseline)
        channel_std.append(channel_std_baseline)
        
        normal_mean_entropy.append(channel_mean_baseline)
        normal_std_entropy.append(channel_std_baseline)
        
    normal_animal_mean.append(np.mean(channel_mean))
    normal_animal_std.append(np.mean(channel_std))
        
    

# In[ ]: HF animals: 1666, 1670, 1690,  1692, 1767, 1774, 1841, 1843
    
root_dir = '../HeartFailureAnimals'
# animals_entropy = [1666, 1767, 1769, 1774, 1841, 1843, 1670, 1692, 1690]
animal_names = os.listdir(root_dir) #HFs

HF_mean_entropy = []
HF_std_entropy = []

HF_animal_mean = []
HF_animal_std = []

channel_entropy_baseline = []
# entropy_filename = 'HFs_entropy_mean.csv'
entropy_filename = 'HFs_entropy_mean.csv'


for animal in animal_names:
    
    if animal == "pig1770" or animal == "pig1844" or animal == "pig1768":
        print('passed this: ', animal)
        continue
    else:
        
        animal_attmetric_dir = root_dir + '/' + animal + '/AttentionMetric_1min_20minbuff/'
        channel_mean = []
        channel_std = []
        
        for channel in sorted(os.listdir(animal_attmetric_dir)):
            
            channel_dir = animal_attmetric_dir + channel + '/'
            
            channel_csv_list = [file for file in glob.glob( channel_dir +"*.csv")]
            
            event_names = [event_filename.split('\\', 10)[-1] for event_filename in channel_csv_list]
    
            #isoalte baseline csv
            event_name = [event_name for event_name in event_names \
                if ("0_shannon" in event_name) and (event_name[event_name.find("0_shannon") - 1] != '1')][0]
                
            print(animal, channel, event_name)
            event_filename = channel_dir + event_name
            
            time, entropy = getEntropy(event_filename)
            
            #rewrite channel name to standardize
            chno=channel.split('_icn')[1]
            chname="channel"+str(chno)
            
            channel_entropy_baseline.append({'animal': animal, 'event_name': event_name, 'channel': chname, 'entropy': np.mean(np.array(entropy))})
            
            d = pd.DataFrame(channel_entropy_baseline)
            d.to_csv(entropy_filename, index=False)
            
            channel_mean_baseline, channel_std_baseline = getMeanStdEntropy(entropy)
            
            channel_mean.append(channel_mean_baseline)
            channel_std.append(channel_std_baseline)
            
            HF_mean_entropy.append(channel_mean_baseline)
            HF_std_entropy.append(channel_std_baseline)
            
        HF_animal_mean.append(np.mean(channel_mean))
        HF_animal_std.append(np.mean(channel_std))    
        
# In[ ]: all entropies are collated in  C:\Users\ngurel\Documents\Stellate_Recording_Files\Data\junk entropy_all.csv      
from statsmodels.stats.anova import AnovaRM

entropy_file = './entropy_all.csv'

df_entropy = pd.read_csv(entropy_file)

print(AnovaRM(data=df_entropy, depvar='entropy_mean', subject='animal', within=['animal_type'], aggregate_func='mean').fit())












