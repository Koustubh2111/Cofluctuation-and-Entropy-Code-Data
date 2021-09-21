# -*- coding: utf-8 -*-
"""
Results with narrowest CIs: (space: 60%,75%,90% exceedance, 40-90 state threshold, zero event rate cases excluded )
excel summary: Bootstrap_combined_EvRate.xlsx
    
"""
# In[ ]: 
from string import ascii_letters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker


# In[ ]: CASE: SpikerateCoact Filepaths
# Normal animals: n=6 
# ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
"""

N1	0p9	60
N2	0p9	90
N3	0p75 90
N4	0p9	90
N5	0p9	90
N6	0p9	90
"""
filepaths_Normal = [                                         
                    '../NormalAnimals/N1/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                    '../NormalAnimals/N2/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                    '../NormalAnimals/N3/SpikerateCoact_output_1min_20minbuff_0p75/coactivity_stats.csv',\
                    '../NormalAnimals/N4/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                    '../NormalAnimals/N5/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                    '../NormalAnimals/N6/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv']
    
state_thresholds_Normal = [60,90,90,90,90,90]  

Normal_path = '../NormalAnimals/'
filenames_Normal = os.listdir(Normal_path)
filenames_Normal = [f for f in filenames_Normal if (f.startswith("N"))]
print(filenames_Normal) 
  
# HF Animals: n=11 
## ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11']
"""
H1	0p9	70
H2	0p9	90
H3	0p75	90
H4	0p9	90
H5	0p9	90
H6	0p9	90
H7	0p9	90
H8	0p9	90
H9	0p9	90
H10	0p9	90
H11	0p9	90
"""    
filepaths_HF = ['../HeartFailureAnimals/H1/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H2/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H3/SpikerateCoact_output_1min_20minbuff_0p75/coactivity_stats.csv',\
                '../HeartFailureAnimals/H4/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H5/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H6/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H7/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H8/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H9/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H10/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H11/SpikerateCoact_output_1min_20minbuff_0p9/coactivity_stats.csv']    

state_thresholds_HF = [70,90,90,90,90,90,90,90,90,90,90] 

HF_path = '../HeartFailureAnimals/'
filenames_HF = os.listdir(HF_path)
filenames_HF = [f for f in filenames_HF if (f.startswith("H"))]
print(filenames_HF) 

#End of Baseline timestamps 
EndBaseline_HF = [15157.47730, 13782.64500, 14479.24235, 15010.85545, 20138.13390, 14126.76400, 22447.50400, 19488.27205, 19001.37350, 16823.12835, 19430.61330]
EndBaseline_Normal = [18081.77015, 14387.14405, 17091.46195, 21465.20400, 28360.64150, 22006.09015]


# In[ ]: CASE: SpikestdCoact Filepaths
# Normal animals: n=6 
# ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
"""
N1	0p75 70
N2	0p9	90
N3	0p9	70
N4	0p9	90
N5	0p9	90
N6	0p9	90
"""
filepaths_Normal = [                                         
                    '../NormalAnimals/N1/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                    '../NormalAnimals/N2/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                    '../NormalAnimals/N3/SpikestdCoact_output_1min_20minbuff_0p75/coactivity_stats.csv',\
                    '../NormalAnimals/N4/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                    '../NormalAnimals/N5/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                    '../NormalAnimals/N6/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv']
    
state_thresholds_Normal = [70,90,70,90,90,90]  

Normal_path = 'C:/Users/ngurel/Documents/Stellate_Recording_Files/Data/NormalAnimals/'
filenames_Normal = os.listdir(Normal_path)
filenames_Normal = [f for f in filenames_Normal if (f.startswith("N"))]
print(filenames_Normal) 
  
# HF Animals: n=11 
## ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11']
"""
H1	0p75	90
H2	0p9	90
H3	0p9	40
H4	0p9	90
H5	0p6	90
H6	0p9	90
H7	0p9	90
H8	0p9	90
H9	0p9	90
H10	0p9	90
H11	0p9	90

"""    
filepaths_HF = ['../HeartFailureAnimals/H1/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H2/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H3/SpikestdCoact_output_1min_20minbuff_0p75/coactivity_stats.csv',\
                '../HeartFailureAnimals/H4/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H5/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H6/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H7/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H8/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H9/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H10/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv',\
                '../HeartFailureAnimals/H11/SpikestdCoact_output_1min_20minbuff_0p9/coactivity_stats.csv']    

state_thresholds_HF = [90,90,40,90,90,90,90,90,90,90,90] 

HF_path = '../HeartFailureAnimals/'
filenames_HF = os.listdir(HF_path)
filenames_HF = [f for f in filenames_HF if (f.startswith("H"))]
print(filenames_HF) 

#End of Baseline timestamps 
EndBaseline_HF = [15157.47730, 13782.64500, 14479.24235, 15010.85545, 20138.13390, 14126.76400, 22447.50400, 19488.27205, 19001.37350, 16823.12835, 19430.61330]
EndBaseline_Normal = [18081.77015, 14387.14405, 17091.46195, 21465.20400, 28360.64150, 22006.09015]



# In[ ]: Normals states

# main figure
fig, ax_Normal = plt.subplots(figsize = (22,12), nrows = len(filepaths_Normal), ncols = 1)
str_normal_state_title= "States for normal animals (optimal exceedance & state threshold pairs)" 
# fig.suptitle(str_normal_state_title, fontsize=16)

count = 0
state_timestamp_Normal = []
EvRate_Normal=[]

for file in filepaths_Normal:

    df = pd.read_csv(file)
    time = df['time']
    stats = df['coactivity_stat']
    
    #time before end of baseline
    index = time < EndBaseline_Normal[count]
    time = time[index]
    stats = stats[index]
    
    #event rate calculation
    
    state_array = np.zeros(len(stats))
              
    for i in range(len(stats)):
        if stats[i] > state_thresholds_Normal[count]:
            state_array[i] = 1
            
    transition_timestamp = []
    for i in range(len(state_array) - 1):
        if (state_array[i] - state_array[i + 1]) == -1:
            transition_timestamp.append([i + 1])
    
    denom = EndBaseline_Normal[count] - time[0] #denominator for calculating event rate
    event_rate = len(transition_timestamp) / (denom)
    EvRate_Normal.append(event_rate)
    state_timestamp_Normal.append(transition_timestamp)
    
    # ax_Normal[count].plot(time/3600, state_array ,'--', color = 'dodgerblue', alpha=0.8)
    ax_Normal[count].plot(time/3600, stats ,'--', color = 'dodgerblue', alpha=0.8)

    #ax_Normal[count].set_xticks(np.array(transition_timestamp)/3600)
    ax_Normal[count].set_xlim(time[0]/3600,EndBaseline_Normal[count]/3600) #limiting to baseline data only
    #ax_Normal[count].axvline(x=EndBaseline_Normal[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
    # ax_Normal[count].tick_params(axis="x", labelsize=3)
    ax_Normal[count].set_xticks([])    
    ax_Normal[count].set_yticks([0,120])     #1 for event
    ax_Normal[count].spines["top"].set_visible(False)  
    ax_Normal[count].spines["right"].set_visible(False)  
    ax_Normal[count].spines["bottom"].set_visible(False)  
    ax_Normal[count].set_ylabel(filenames_Normal[count], fontsize=16)
    ax_Normal[count].set_ylabel(count + 1, fontsize=16)

    count = count + 1
    
# ax_Normal[count-1].set_xlabel('Baseline time (hours)', fontsize=16)


# str_normals_state_savefig_pdf= "Normals_SpikerateCoact_Opt_pairs_base_supp.pdf"
# str_normals_state_savefig_pdf= "Normals_SpikerateCoact_Opt_pairs_base_cofluctuation_supp.pdf"

# str_normals_state_savefig_pdf= "Normals_SpikestdCoact_Opt_pairs_base_supp.pdf"
str_normals_state_savefig_pdf= "Normals_SpikestdCoact_Opt_pairs_base_cofluctuation_supp.pdf"

plt.savefig(str_normals_state_savefig_pdf)


# In[ ]: HFs states

# main figure
fig, ax_HF = plt.subplots(figsize = (22,12), nrows = len(filepaths_HF), ncols = 1)
str_HF_state_title= "States for HF animals (optimal exceedance & state threshold pairs)" 
# fig.suptitle(str_HF_state_title, fontsize=16)

count = 0
state_timestamp_HF = []
EvRate_HF=[]
# time=[]
# stats = []
# index = []
for file in filepaths_HF:

    df = pd.read_csv(file)
    time = df['time']
    stats = df['coactivity_stat']
    
    #time before end of baseline
    index = time < EndBaseline_HF[count]
    time = time[index]
    stats = stats[index]
    
    #event rate calculation
    
    state_array = np.zeros(len(stats))
              
    for i in range(len(stats)):
        if stats[i] > state_thresholds_HF[count]:
            state_array[i] = 1
            
    transition_timestamp = []
    for i in range(len(state_array) - 1):
        if (state_array[i] - state_array[i + 1]) == -1:
            transition_timestamp.append([i + 1])
    
    denom = EndBaseline_HF[count] - time[0] #denominator for calculating event rate
    event_rate = len(transition_timestamp) / (denom)
    EvRate_HF.append(event_rate)
    state_timestamp_HF.append(transition_timestamp)
    
    # ax_HF[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
    ax_HF[count].plot(time/3600, stats ,'--', color = 'orchid', alpha=0.8)
    #ax_HF[count].set_xticks(np.array(transition_timestamp)/3600)
    ax_HF[count].set_xlim(time[0]/3600,EndBaseline_HF[count]/3600) #limiting to baseline data only
    #ax_HF[count].axvline(x=EndBaseline_HF[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
    #ax_HF[count].tick_params(axis="x", labelsize=3)
    ax_HF[count].set_xticks([])  
    ax_HF[count].set_yticks([0,120])    #1 for er
    ax_HF[count].spines["top"].set_visible(False)  
    ax_HF[count].spines["right"].set_visible(False)  
    ax_HF[count].spines["bottom"].set_visible(False)  
    # ax_HF[count].set_ylabel(filenames_HF[count], fontsize=10)
    ax_HF[count].set_ylabel(count + 1, fontsize=10)

    count = count + 1
    
#ax_HF[count-1].set_xlabel('Baseline time (hours)', fontsize=16)


# str_HFs_state_savefig_pdf= "HFs_SpikerateCoact_final_pairs_base_supp.pdf"
# str_HFs_state_savefig_pdf= "HFs_SpikerateCoact_final_pairs_base_cofluctuations_supp.pdf"

# str_HFs_state_savefig_pdf= "HFs_SpikestdCoact_final_pairs_base_supp.pdf"
str_HFs_state_savefig_pdf= "HFs_SpikestdCoact_final_pairs_base_cofluctuations_supp.pdf"

plt.savefig(str_HFs_state_savefig_pdf)

