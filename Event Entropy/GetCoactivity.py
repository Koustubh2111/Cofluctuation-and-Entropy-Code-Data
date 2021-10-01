#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:07:03 2021

@author: koustubh
"""

#JOBS:
#choose the right folderwith the chosen threshold
#get coactivity stats csv file for that animal
import numpy as np
import pandas as pd
from random import choices #Sampling with replacement 
def getArrayBaseline(filename, baseline):
    
    #Takes file and extracts baseline as time and stats
    df = pd.read_csv(filename)
    
    time = df['time']
    stats = df['coactivity_stat']
    
    #IndexFrEndOfBasline
    index = time <= baseline
    time = time[index]
    stats = stats[index]
    
    stats = np.array(stats)
    time = np.array(time)
    
    return stats, time

def ResampleSeries(series, num):
    
    #using sampling with replacement on length of array
    return choices(series, k = num)

def getStartStopEventTimeStamps(time, stats, threshold, toggle = True):
    
    #time and series  - the coactivity time series, mean or std
    #threshold - 70 or 10
    #toggle - true : greater than; toggle - false : lesser
    
    
    state_array = np.zeros(len(stats))
    for i in range(len(stats)):
        if toggle:
            if stats[i] > 70:
                state_array[i] = 1
        else:
            if stats[i] < 10:
                state_array[i] = 1

            
    transition_stamps_start = []
    for i in range(len(state_array) - 1):
        if (state_array[i] - state_array[i + 1]) == -1:
            transition_stamps_start.append(time[i + 1])
            
    transition_stamps_end = []
    for i in range(len(state_array) - 1):
        if (state_array[i] - state_array[i + 1]) == 1:
            transition_stamps_end.append(time[i + 1])      

    return transition_stamps_start, transition_stamps_end
    

def getEventRateFromSeries(time, stats, threshold):
    
    #Takes the coact array after baseline extraction and returns event rate in baseline
    #Get state array, time stamp array and event rate
    #threshold- 40%, 50%, 60%, 70% etc
    
    state_array = np.zeros(len(stats))
    for i in range(len(stats)):
        if stats[i] > threshold:
            state_array[i] = 1
            
    transition_stamps = []
    for i in range(len(state_array) - 1):
        if (state_array[i] - state_array[i + 1]) == -1:
            transition_stamps.append(time[i + 1])
    
    if len(transition_stamps) > 0:
        return len(transition_stamps) / (transition_stamps[-1] - transition_stamps[0])
    else:
        return 10e-10
    
    
    
