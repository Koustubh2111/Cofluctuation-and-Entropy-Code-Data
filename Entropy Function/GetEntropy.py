#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:56:58 2021

@author: koustubh
"""
import numpy as np
import pandas as pd
from operator import add

def getEntropy(filename):
        
    df = pd.read_csv(filename)
    time = df['time']
    entropy = df['shannon_entropy']
    
    #The above entropy has repeated values and -999
    
    #Remove -999
    index = entropy > 0
    time = time[index]
    entropy = entropy[index]
    
    #Remove repeated values
    filtered_entropy_index = np.nonzero(np.abs(np.diff(entropy)))[0]
    
    #Convert to entropy to array to prevent depreciation
    entropy = np.array(entropy)
    time = np.array(time)
    
    entropy = entropy[filtered_entropy_index]
    time = time[filtered_entropy_index]    
    
    return time, entropy

def getRawDataFromHist(hist, num_bins):
    
    bins = np.linspace(0,1,num_bins)
    raw_data = []
    
    for bin, count in zip(bins, hist):
        raw_data.extend( list( np.repeat(bin, count) ) )
    
    return raw_data 
    
    
    
    
    
def getHist(entropy, num_bins):
    
    #returns a hist of entropy
    
    hist, bins = np.histogram(entropy, bins = num_bins, range = (0,0.8), density = True)
    return hist
    
def getMeanHist(hist_list):
    
    #https://stackoverflow.com/questions/18713321/element-wise-addition-of-2-lists
    num_hist = len(hist_list)
    if num_hist == 0:
        return None
    #print(num_hist)
    final_list = hist_list[0] #adding to final_list every iteration
    for hist in hist_list:        
        final_list = map(add, final_list, hist)        
        
    final_list = [f/num_hist for f in final_list] 
    return final_list

def getSeriesFromWindow(time, series, start, end):
    
    index = np.logical_and(time >= start, time <= end)
    return series[index]
    
