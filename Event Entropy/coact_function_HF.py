#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: koustubh
"""
final_dataset = []

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from GetCoactivity import getArrayBaseline, getEventRateFromSeries, ResampleSeries, getStartStopEventTimeStamps
from GetEntropy import getEntropy, getHist, getMeanHist, getRawDataFromHist
import glob
from scipy.stats import ks_2samp
import seaborn as sns
from statannot import add_stat_annotation


HF_dir = '../HeartFailureAnimals/' #Entire Directory available upon request; complet ata not available in the repo
HF_animal_list = os.listdir(HF_dir)[1:-4]
EndBaseline = [15157.47730, '', 13782.64500, 14479.24235, 15010.85545, 20138.13390, '', '', '', 19488.27205, 19001.37350, \
               16823.12835, '']
  

    
label = ['P', \
         '',\
        'R', \
        'R', \
        'R', \
        'P', \
        '', \
        '', \
        '', \
        'P', \
        'P', \
        'P', \
        '']
    
animal_exceedance = [9, '', 9, 75, 9, 9, '', '', '', 9, 9, 9, '']
animal_thresholds = [70, '', 90, 90, 90, 90, '', '', '', 90, 90, 90, '']

if len(HF_animal_list) == len(EndBaseline) == len(label):
    pass
else:
    print('Cannot zip unequal lengths')
    sys.exit()
num_bins = 25

filtered_animal_list = []
animal_entropy_event_rate = []
animal_entropy_non_event_rate = []
animal_entropy_event_std = []
animal_entropy_non_event_std = []
for hf_animal, baseline, animal_label, ex, th in zip(HF_animal_list, EndBaseline, \
                                                     label, animal_exceedance, animal_thresholds):
    
    if animal_label == '':
        continue
    
    
    animal_dir = HF_dir + hf_animal + '/'
    ########################## RATE COACT #####################################
    #Coact on mean
    coact_mean_file = animal_dir + 'SpikerateCoact_output_1min_20minbuff_0p' + str(ex) + '/coactivity_stats.csv'
    
    #Get stats and time
    stats, time  = getArrayBaseline(coact_mean_file, baseline)
    
    #Get start and stop timestamps for the 70 and 10 threshold of the coactivity mean time series
    transition_stamps_start_event_rate, transition_stamps_end_event_rate = getStartStopEventTimeStamps(time, stats, th, toggle = True)
    
    ############################# STD COACT ###################################
    coact_std_file = animal_dir + 'SpikestdCoact_output_1min_20minbuff_0p' + str(ex) + '/coactivity_stats.csv'
    
    #Get stats and time
    stats, time = getArrayBaseline(coact_std_file, baseline)       
        
    #Get start and stop timestamps for the 70 and 10 threshold of the coactivity std time series
    transition_stamps_start_event_std, transition_stamps_end_event_std = getStartStopEventTimeStamps(time, stats, th, toggle = True)
    
    #skip channel if one of the timestamp is empty
    if len(transition_stamps_start_event_rate) == 0 or\
       len(transition_stamps_end_event_rate) == 0 or\
       len(transition_stamps_start_event_std) == 0 or\
       len(transition_stamps_end_event_std) == 0 :
           
           continue
    
    #Force timestamps RATE
    if (len(transition_stamps_start_event_rate) != len(transition_stamps_end_event_rate) ) \
            and ( transition_stamps_end_event_rate[0] < transition_stamps_start_event_rate[0] ):
                
                transition_stamps_end_event_rate = transition_stamps_end_event_rate[1:]               
    
    
    #Force timestamps STD
    if (len(transition_stamps_start_event_std) != len(transition_stamps_end_event_std) ) \
            and ( transition_stamps_end_event_std[0] < transition_stamps_start_event_std[0] ):
                
                transition_stamps_end_event_std = transition_stamps_end_event_std[1:]    
                
    ################ ENTROPY #########################################################                
    entropy_channel_list = os.listdir(animal_dir + 'AttentionMetric_1min_20minbuff/')
    
    channel_entropy_events_rate = []
    channel_entropy_non_events_rate = []
    channel_entropy_events_std = []
    channel_entropy_non_events_std = []
    for channel in entropy_channel_list:
            
        #Get list of .csv files in the channel directory
        channel_dir = animal_dir + 'AttentionMetric_1min_20minbuff/' + channel + '/'
        channel_csv_list = [file for file in glob.glob( channel_dir +"*.csv")]
        
        #isolate the event names : SAFETY
        event_name = [csv_file.split('/', 10)[-1] for csv_file in channel_csv_list]
        
        #Isolate the file ending with 0 for baseline
        baseline_file = [event for event in event_name if ("0_shannon" in event) and\
            (event[event.find("0_shannon") - 1] != '1')]            
        baseline_file = baseline_file[0]               
        
        #Get Entrop of channel
        time, entropy = getEntropy(channel_dir + baseline_file)
        
        #extract entropy under coact rate event timestamps
        #EVENTS
        event_entropy = []
        for start, end in zip(transition_stamps_start_event_rate, transition_stamps_end_event_rate):
                    
            index = np.logical_and(time >= start, time <= end)
            if len(entropy[index]) == 0:
                continue
            
            channel_entropy_events_rate.extend(entropy[index])
            event_entropy.extend(entropy[index])
            
        
            
            
        #NON EVENTS
        non_event_entropy = []
        non_event_entropy.extend(entropy[ time < transition_stamps_start_event_rate[0] ])
        channel_entropy_non_events_rate.extend( entropy[ time < transition_stamps_start_event_rate[0] ] )
        for end, next_start in zip(transition_stamps_end_event_rate[:-1], transition_stamps_start_event_rate[1:]):
            
            index = np.logical_and(time >= end, time <= next_start)
            if len(entropy[index]) == 0:
                continue
            channel_entropy_non_events_rate.extend(entropy[index])
            non_event_entropy.extend(entropy[index])
        
        channel_entropy_non_events_rate.extend( entropy[ time > transition_stamps_end_event_rate[-1]])
        non_event_entropy.extend(entropy[ time > transition_stamps_end_event_rate[-1]])
        
        final_dataset.append([hf_animal, channel, np.mean(entropy), np.mean(event_entropy), \
                       np.std(entropy), np.std(event_entropy), 'Event', 'Coact_Mean'])
            
        final_dataset.append([hf_animal, channel, np.mean(entropy), np.mean(non_event_entropy), \
                       np.std(entropy), np.std(non_event_entropy), 'Non Event', 'Coact_Mean'])
        
        
        
        #extract entropy under coact std event timestamps
        #EVENTS
        event_entropy = []
        for start, end in zip(transition_stamps_start_event_std, transition_stamps_end_event_std):
                    
            index = np.logical_and(time >= start, time <= end)
            if len(entropy[index]) == 0:
                continue
            
            channel_entropy_events_std.extend(entropy[index])
            event_entropy.extend(entropy[index])
            
        #NON EVENTS
        non_event_entropy = []
        non_event_entropy.extend(entropy[ time < transition_stamps_start_event_std[0] ] )
        channel_entropy_non_events_std.extend( entropy[ time < transition_stamps_start_event_std[0] ] )
        for end, next_start in zip(transition_stamps_end_event_std[:-1], transition_stamps_start_event_std[1:]):
            
            index = np.logical_and(time >= end, time <= next_start)
            if len(entropy[index]) == 0:
                continue
            channel_entropy_non_events_std.extend(entropy[index])
            non_event_entropy.extend(entropy[index])
        
        channel_entropy_non_events_std.extend( entropy[ time > transition_stamps_end_event_std[-1]])
        non_event_entropy.extend(entropy[ time > transition_stamps_end_event_std[-1]])
        
        final_dataset.append([hf_animal, channel, np.mean(entropy), np.mean(event_entropy), \
                       np.std(entropy), np.std(event_entropy), 'Event', 'Coact_STD'])
            
        final_dataset.append([hf_animal, channel, np.mean(entropy), np.mean(non_event_entropy), \
                       np.std(entropy), np.std(non_event_entropy), 'Non Event', 'Coact_STD'])
        
        
    animal_entropy_event_rate.append(channel_entropy_events_rate)
    animal_entropy_non_event_rate.append(channel_entropy_non_events_rate)
    animal_entropy_event_std.append(channel_entropy_events_std)
    animal_entropy_non_event_std.append(channel_entropy_non_events_std)
    filtered_animal_list.append([hf_animal, animal_label])
    
columns = ['Animal', 'Channel', 'Baseline_Entropy_Mean', 'Event/NonEvent_Entropy_Mean', \
           'Baseline_Entropy_Std', 'Event/NonEvent_Entropy_Std', 'Event/Nonevent', 'Coact_Type']
pd.DataFrame(final_dataset, columns = columns, index = None).to_csv('heart_failure_coact_function.csv')
#hue must be events and non events    
#x must be rate or std timestamps
#y must be entropy
"""
#per animal figure
for event_rate, non_event_rate, event_std, non_event_std, animal_name in zip(animal_entropy_event_rate, animal_entropy_non_event_rate,\
    animal_entropy_event_std, animal_entropy_non_event_std, filtered_animal_list):
    
    #Animal violin plot
    fig, ax = plt.subplots(figsize = (20,10), nrows = 1, ncols = 2)
    fig.tight_layout(pad = 4.0)
    
    df_er = pd.DataFrame(event_rate, columns = ['Event'])
    df_ner = pd.DataFrame(non_event_rate, columns = ['Non Event'])
    df1 = pd.concat([df_er, df_ner], axis=1)
   
    sns.violinplot(data = df1, ax = ax[0])
    
    #ks_stat, pval = ks_2samp( list(df1['Event']), list(df1['Non Event']) )
    #title_str = 'KS Stat : ' + str(ks_stat) + ' P val : ' + str(pval) + ' ' + animal_name[1]
    
    add_stat_annotation(ax[0], data=df1,
                                   box_pairs=[('Event', 'Non Event')],
                                   test='Mann-Whitney', text_format='simple',
                                   verbose=2, comparisons_correction = None)
    
    ax[0].set_title('Coact rate timestamps', fontsize = 14)
    ax[0].tick_params(labelsize = 14)
    ax[0].set_ylim([0,1])
    
    df_es = pd.DataFrame(event_std, columns = ['Event'])
    df_nes = pd.DataFrame(non_event_std, columns = ['Non Event'])
    df2 = pd.concat([df_es, df_nes], axis=1)  
    sns.violinplot(data = df2,  ax = ax[1])
    
    add_stat_annotation(ax[1], data=df2,
                                   box_pairs=[('Event', 'Non Event')],
                                   test='Mann-Whitney', text_format='simple',
                                   verbose=2, comparisons_correction = None)
    
    #ks_stat, pval = ks_2samp( list(df2['Event']), list(df2['Non Event']) )
    #title_str = 'KS Stat : ' + str(ks_stat) + ' P val : ' + str(pval) + ' ' + animal_name[1]
    
    ax[1].set_title('Coact std timestamps', fontsize = 14)
    ax[1].tick_params(labelsize = 14)
    ax[1].set_ylim([0,1]) 

    plt.savefig(animal_name[0] + 'coact_fn.pdf')               
    
"""           
    
