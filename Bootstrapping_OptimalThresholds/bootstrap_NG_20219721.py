# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 18:18:50 2021

@author: NGurel

bootstrapping based on event  rate on one animal

num_bs_replicates=50000 takes too much time, I did 1000 instead.

Results with narrowest CIs: (space: 60%,75%,90% exceedance, 40-90 state threshold )

CASE: SpikerateCoact			
animal	min of exceedance & state threshold combination	winner exceedance	winner state_threshold
pig1666	0p9	70
pig1670	0p9	90
pig1690pvccmrtx	0p75	90
pig1692chronicPVCRTX	0p9	90
pig1767pvc	0p9	90
pig1768	0p9	90
pig1770	0p9	90
pig1774pvc	0p9	90
pig1841pvc	0p9	90
pig1843pvc	0p9	90
pig1844	0p9	90
pig1720	0p9	60
pig1721	0p9	90
pig1723	0p9	80
pig1740	0p9	90
pig1741	0p9	90
pig1742	0p9	90

CASE: SpikestdCoact			
animal	min of exceedance & state threshold combination	winner exceedance	winner state_threshold
pig1666	0p9	90
pig1670	0p9	90
pig1690pvccmrtx	0p9	90
pig1692chronicPVCRTX	0p9	90
pig1767pvc	0p9	60
pig1768	0p9	90
pig1770	0p9	90
pig1774pvc	0p9	90
pig1841pvc	0p9	90
pig1843pvc	0p9	90
pig1844	0p9	90
pig1720	0p9	90
pig1721	0p9	90
pig1723	0p9	90
pig1740	0p9	90
pig1741	0p9	90
pig1742	0p9	90
    

"""

# In[ ]: 
from string import ascii_letters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# In[ ]: fns

def getResampledStats(time, stats):
    resampled_index = np.random.choice(np.arange(len(time)), len(time))
    print(resampled_index)
    return np.array(time)[resampled_index], np.array(stats)[resampled_index]

    
def getEventRate(denom, stats,threshold):
    #time - resampled time
    #stats - resample stats with same index as resampled time
    state_array = np.zeros(len(stats))
              
    for i in range(len(stats)):
        if stats[i] > threshold:
            state_array[i] = 1
            
    transition_timestamp = []
    for i in range(len(state_array) - 1):
        if (state_array[i] - state_array[i + 1]) == -1:
            transition_timestamp.append([i + 1])
            
    return len(transition_timestamp) / (denom)

def draw_bs_replicates(denom,time,stats,size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        #bs_sample = np.random.choice(data,size=len(data))
        _, bb = getResampledStats(time, stats)
        rate = getEventRate(denom,bb,threshold)
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = rate
        
    return bs_replicates


# In[ ]: EXAMPLE ONE ANIMAL
df = pd.read_csv('C:/Users/ngurel/Documents/Stellate_Recording_Files/Data/HeartFailureAnimals/pig1844/SpikerateCoact_output_1min_20minbuff_0p6/coactivity_stats.csv')
time = df['time']
stats = df['coactivity_stat']

endbaseline_1844 = 19430.61330

threshold = 50 
num_bs_replicates=100 #Change to 50000

#time before end of baseline
index = time < endbaseline_1844
time = time[index]
stats = stats[index]

#convert to lists
time = time.tolist()
stats = stats.tolist()


if len(stats) > 0:
    # Draw N bootstrap replicates
    denom = time[-1] - time[0] #hard coded
    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
    
    # Print empirical mean
    #print("Empirical mean: " + str(np.mean(values)))
    
    # Print the mean of bootstrap replicates
    #print("Bootstrap replicates mean: " + str(np.mean(bs_replicates_values)))

    ########################### COMMENT IF PLOTTING STATES  ###################################################### 

    # Plot the PDF for bootstrap replicates as histogram & save fig
    plt.hist(bs_replicates_values,bins=30)
    
    lower=5
    upper=95
    # Showing the related percentiles
    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
    
    plt.xlabel("Event rate")
    plt.ylabel("Probability Density Function")
    #plt.title("pig" + current_animal + " SpikerateCoact_output_1min_20minbuff_0p6" +" Th: " + str(threshold))
    plt.legend()
    #str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikerateCoact_output_1min_20minbuff_0p6" + "_Threshold" +  str(threshold) + "_bootstrap" + str(num_bs_replicates) + "_baseline.pdf"
    #plt.savefig(str_PDF_savefig_pdf)    
    plt.show()
    
    # Get the corresponding values of 5th and 95th CI
    CI_BS = np.percentile(bs_replicates_values,[lower,upper])
    CI_width = np.diff(CI_BS)
    # Print stuff
    print("event rate replicates: ",bs_replicates_values)
    print("event rate replicates mean: ",np.mean(bs_replicates_values))
    print("event rate replicates std: ",np.std(bs_replicates_values))
    print("The confidence interval: ",CI_BS)
    print("CI width: ",CI_width)
       
else:
    print("has no transition timestamps for threshold = " + str(threshold))


# In[ ]:calculation params


#End of Baseline timestamps 
EndBaseline_HF = [15157.47730, 13782.64500, 14479.24235, 15010.85545, 20138.13390, 14126.76400, 22447.50400, 19488.27205, 19001.37350, 16823.12835, 19430.61330]
EndBaseline_Normal = [18081.77015, 14387.14405, 17091.46195, 21465.20400, 28360.64150, 22006.09015]

#End of Baseline linewidth
lw_EndBaseline = 3

# In[ ]: Data: HF Animals

HF_path = 'C:/Users/ngurel/Documents/Stellate_Recording_Files/Data/HeartFailureAnimals/'
filenames = os.listdir(HF_path)
filenames = [f for f in filenames if (f.startswith("pig"))]

print(filenames)
# ['pig1666', 'pig1670', 'pig1690pvccmrtx', 'pig1692chronicPVCRTX', 'pig1767pvc', 'pig1768', 'pig1770', 'pig1774pvc', 'pig1841pvc', 'pig1843pvc', 'pig1844']

# In[ ]: SpikerateCoact_output_1min_20minbuff_0p6 : EACH In[] AFTER THIS ONE IS REPEAT 

threshold = 10 
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_HF = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_HF = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_HF_state_title= "States for HF animals from SpikerateCoact_output_1min_20minbuff_0p6, threshold = " + str(threshold)
# fig.suptitle(str_HF_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(HF_path, filename)
    current_path_SpikerateCoact_output_1min_20minbuff_0p6 = os.path.join(current_path, 'SpikerateCoact_output_1min_20minbuff_0p6').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikerateCoact_output_1min_20minbuff_0p6):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikerateCoact_output_1min_20minbuff_0p6, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_HF[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_HF[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_HF[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_HF[count].set_xlim(time[0]/3600,EndBaseline_HF[count]/3600) #limiting to baseline data only
                # #ax_HF[count].axvline(x=EndBaseline_HF[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_HF[count].tick_params(axis="x", labelsize=3)
                # ax_HF[count].set_yticks([0,1])    
                # ax_HF[count].spines["top"].set_visible(False)  
                # ax_HF[count].spines["right"].set_visible(False)  
                # ax_HF[count].spines["bottom"].set_visible(False)  
                # ax_HF[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikerateCoact_output_1min_20minbuff_0p6" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikerateCoact_output_1min_20minbuff_0p6" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikerateCoact_output_1min_20minbuff_0p6"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_HF_SpikerateCoact_output_1min_20minbuff_0p6" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 
     
# In[ ]: SpikerateCoact_output_1min_20minbuff_0p75

#threshold = 40
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_HF = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_HF = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_HF_state_title= "States for HF animals from SpikerateCoact_output_1min_20minbuff_0p75, threshold = " + str(threshold)
# fig.suptitle(str_HF_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(HF_path, filename)
    current_path_SpikerateCoact_output_1min_20minbuff_0p75 = os.path.join(current_path, 'SpikerateCoact_output_1min_20minbuff_0p75').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikerateCoact_output_1min_20minbuff_0p75):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikerateCoact_output_1min_20minbuff_0p75, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_HF[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_HF[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_HF[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_HF[count].set_xlim(time[0]/3600,EndBaseline_HF[count]/3600) #limiting to baseline data only
                # #ax_HF[count].axvline(x=EndBaseline_HF[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_HF[count].tick_params(axis="x", labelsize=3)
                # ax_HF[count].set_yticks([0,1])    
                # ax_HF[count].spines["top"].set_visible(False)  
                # ax_HF[count].spines["right"].set_visible(False)  
                # ax_HF[count].spines["bottom"].set_visible(False)  
                # ax_HF[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikerateCoact_output_1min_20minbuff_0p75" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikerateCoact_output_1min_20minbuff_0p75" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikerateCoact_output_1min_20minbuff_0p75"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_HF_SpikerateCoact_output_1min_20minbuff_0p75" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 

# In[ ]: SpikerateCoact_output_1min_20minbuff_0p9 

#threshold = 50
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_HF = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_HF = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_HF_state_title= "States for HF animals from SpikerateCoact_output_1min_20minbuff_0p9, threshold = " + str(threshold)
# fig.suptitle(str_HF_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(HF_path, filename)
    current_path_SpikerateCoact_output_1min_20minbuff_0p9 = os.path.join(current_path, 'SpikerateCoact_output_1min_20minbuff_0p9').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikerateCoact_output_1min_20minbuff_0p9):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikerateCoact_output_1min_20minbuff_0p9, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_HF[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_HF[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_HF[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_HF[count].set_xlim(time[0]/3600,EndBaseline_HF[count]/3600) #limiting to baseline data only
                # #ax_HF[count].axvline(x=EndBaseline_HF[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_HF[count].tick_params(axis="x", labelsize=3)
                # ax_HF[count].set_yticks([0,1])    
                # ax_HF[count].spines["top"].set_visible(False)  
                # ax_HF[count].spines["right"].set_visible(False)  
                # ax_HF[count].spines["bottom"].set_visible(False)  
                # ax_HF[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikerateCoact_output_1min_20minbuff_0p9" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikerateCoact_output_1min_20minbuff_0p9" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikerateCoact_output_1min_20minbuff_0p9"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_HF_SpikerateCoact_output_1min_20minbuff_0p9" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 

# In[ ]: SpikestdCoact_output_1min_20minbuff_0p6 

#threshold = 50
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_HF = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_HF = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_HF_state_title= "States for HF animals from SpikestdCoact_output_1min_20minbuff_0p6, threshold = " + str(threshold)
# fig.suptitle(str_HF_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(HF_path, filename)
    current_path_SpikestdCoact_output_1min_20minbuff_0p6 = os.path.join(current_path, 'SpikestdCoact_output_1min_20minbuff_0p6').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikestdCoact_output_1min_20minbuff_0p6):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikestdCoact_output_1min_20minbuff_0p6, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_HF[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_HF[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_HF[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_HF[count].set_xlim(time[0]/3600,EndBaseline_HF[count]/3600) #limiting to baseline data only
                # #ax_HF[count].axvline(x=EndBaseline_HF[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_HF[count].tick_params(axis="x", labelsize=3)
                # ax_HF[count].set_yticks([0,1])    
                # ax_HF[count].spines["top"].set_visible(False)  
                # ax_HF[count].spines["right"].set_visible(False)  
                # ax_HF[count].spines["bottom"].set_visible(False)  
                # ax_HF[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikestdCoact_output_1min_20minbuff_0p6" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikestdCoact_output_1min_20minbuff_0p6" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikestdCoact_output_1min_20minbuff_0p6"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_HF_SpikestdCoact_output_1min_20minbuff_0p6" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 
     
# In[ ]: SpikestdCoact_output_1min_20minbuff_0p75

#threshold = 50
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_HF = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_HF = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_HF_state_title= "States for HF animals from SpikestdCoact_output_1min_20minbuff_0p75, threshold = " + str(threshold)
# fig.suptitle(str_HF_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(HF_path, filename)
    current_path_SpikestdCoact_output_1min_20minbuff_0p75 = os.path.join(current_path, 'SpikestdCoact_output_1min_20minbuff_0p75').replace("\\","/")

    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikestdCoact_output_1min_20minbuff_0p75):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikestdCoact_output_1min_20minbuff_0p75, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_HF[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_HF[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_HF[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_HF[count].set_xlim(time[0]/3600,EndBaseline_HF[count]/3600) #limiting to baseline data only
                # #ax_HF[count].axvline(x=EndBaseline_HF[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_HF[count].tick_params(axis="x", labelsize=3)
                # ax_HF[count].set_yticks([0,1])    
                # ax_HF[count].spines["top"].set_visible(False)  
                # ax_HF[count].spines["right"].set_visible(False)  
                # ax_HF[count].spines["bottom"].set_visible(False)  
                # ax_HF[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikestdCoact_output_1min_20minbuff_0p75" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikestdCoact_output_1min_20minbuff_0p75" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikestdCoact_output_1min_20minbuff_0p75"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_HF_SpikestdCoact_output_1min_20minbuff_0p75" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 

# In[ ]: SpikestdCoact_output_1min_20minbuff_0p9 

#threshold = 50
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_HF = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_HF = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_HF_state_title= "States for HF animals from SpikestdCoact_output_1min_20minbuff_0p9, threshold = " + str(threshold)
# fig.suptitle(str_HF_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(HF_path, filename)
    current_path_SpikestdCoact_output_1min_20minbuff_0p9 = os.path.join(current_path, 'SpikestdCoact_output_1min_20minbuff_0p9').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikestdCoact_output_1min_20minbuff_0p9):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikestdCoact_output_1min_20minbuff_0p9, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_HF[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_HF[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_HF[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_HF[count].set_xlim(time[0]/3600,EndBaseline_HF[count]/3600) #limiting to baseline data only
                # #ax_HF[count].axvline(x=EndBaseline_HF[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_HF[count].tick_params(axis="x", labelsize=3)
                # ax_HF[count].set_yticks([0,1])    
                # ax_HF[count].spines["top"].set_visible(False)  
                # ax_HF[count].spines["right"].set_visible(False)  
                # ax_HF[count].spines["bottom"].set_visible(False)  
                # ax_HF[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikestdCoact_output_1min_20minbuff_0p9" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikestdCoact_output_1min_20minbuff_0p9" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikestdCoact_output_1min_20minbuff_0p9"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_HF_SpikestdCoact_output_1min_20minbuff_0p9" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 


# In[ ]:  Normal animals
    
Normal_path = 'C:/Users/ngurel/Documents/Stellate_Recording_Files/Data/NormalAnimals/'
filenames = os.listdir(Normal_path)
filenames = [f for f in filenames if (f.startswith("pig"))]
print(filenames) 

# ['pig1720', 'pig1721', 'pig1723', 'pig1740', 'pig1741', 'pig1742']


# In[ ]: SpikerateCoact_output_1min_20minbuff_0p6 : EACH In[] AFTER THIS ONE IS REPEAT 

threshold = 10
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_Normal = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_Normal = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_Normal_state_title= "States for Normal animals from SpikerateCoact_output_1min_20minbuff_0p6, threshold = " + str(threshold)
# fig.suptitle(str_Normal_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(Normal_path, filename)
    current_path_SpikerateCoact_output_1min_20minbuff_0p6 = os.path.join(current_path, 'SpikerateCoact_output_1min_20minbuff_0p6').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikerateCoact_output_1min_20minbuff_0p6):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikerateCoact_output_1min_20minbuff_0p6, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_Normal[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_Normal[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_Normal[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_Normal[count].set_xlim(time[0]/3600,EndBaseline_Normal[count]/3600) #limiting to baseline data only
                # #ax_Normal[count].axvline(x=EndBaseline_Normal[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_Normal[count].tick_params(axis="x", labelsize=3)
                # ax_Normal[count].set_yticks([0,1])    
                # ax_Normal[count].spines["top"].set_visible(False)  
                # ax_Normal[count].spines["right"].set_visible(False)  
                # ax_Normal[count].spines["bottom"].set_visible(False)  
                # ax_Normal[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikerateCoact_output_1min_20minbuff_0p6" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikerateCoact_output_1min_20minbuff_0p6" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikerateCoact_output_1min_20minbuff_0p6"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_Normal_SpikerateCoact_output_1min_20minbuff_0p6" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 
     
# In[ ]: SpikerateCoact_output_1min_20minbuff_0p75

#threshold = 60
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_Normal = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_Normal = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_Normal_state_title= "States for Normal animals from SpikerateCoact_output_1min_20minbuff_0p75, threshold = " + str(threshold)
# fig.suptitle(str_Normal_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(Normal_path, filename)
    current_path_SpikerateCoact_output_1min_20minbuff_0p75 = os.path.join(current_path, 'SpikerateCoact_output_1min_20minbuff_0p75').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikerateCoact_output_1min_20minbuff_0p75):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikerateCoact_output_1min_20minbuff_0p75, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_Normal[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_Normal[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_Normal[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_Normal[count].set_xlim(time[0]/3600,EndBaseline_Normal[count]/3600) #limiting to baseline data only
                # #ax_Normal[count].axvline(x=EndBaseline_Normal[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_Normal[count].tick_params(axis="x", labelsize=3)
                # ax_Normal[count].set_yticks([0,1])    
                # ax_Normal[count].spines["top"].set_visible(False)  
                # ax_Normal[count].spines["right"].set_visible(False)  
                # ax_Normal[count].spines["bottom"].set_visible(False)  
                # ax_Normal[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikerateCoact_output_1min_20minbuff_0p75" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikerateCoact_output_1min_20minbuff_0p75" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikerateCoact_output_1min_20minbuff_0p75"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_Normal_SpikerateCoact_output_1min_20minbuff_0p75" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 

# In[ ]: SpikerateCoact_output_1min_20minbuff_0p9 

#threshold = 50
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_Normal = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_Normal = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_Normal_state_title= "States for Normal animals from SpikerateCoact_output_1min_20minbuff_0p9, threshold = " + str(threshold)
# fig.suptitle(str_Normal_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(Normal_path, filename)
    current_path_SpikerateCoact_output_1min_20minbuff_0p9 = os.path.join(current_path, 'SpikerateCoact_output_1min_20minbuff_0p9').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikerateCoact_output_1min_20minbuff_0p9):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikerateCoact_output_1min_20minbuff_0p9, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_Normal[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_Normal[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_Normal[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_Normal[count].set_xlim(time[0]/3600,EndBaseline_Normal[count]/3600) #limiting to baseline data only
                # #ax_Normal[count].axvline(x=EndBaseline_Normal[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_Normal[count].tick_params(axis="x", labelsize=3)
                # ax_Normal[count].set_yticks([0,1])    
                # ax_Normal[count].spines["top"].set_visible(False)  
                # ax_Normal[count].spines["right"].set_visible(False)  
                # ax_Normal[count].spines["bottom"].set_visible(False)  
                # ax_Normal[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikerateCoact_output_1min_20minbuff_0p9" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikerateCoact_output_1min_20minbuff_0p9" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikerateCoact_output_1min_20minbuff_0p9"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_Normal_SpikerateCoact_output_1min_20minbuff_0p9" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 

# In[ ]: SpikestdCoact_output_1min_20minbuff_0p6 

#threshold = 50
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_Normal = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_Normal = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_Normal_state_title= "States for Normal animals from SpikestdCoact_output_1min_20minbuff_0p6, threshold = " + str(threshold)
# fig.suptitle(str_Normal_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(Normal_path, filename)
    current_path_SpikestdCoact_output_1min_20minbuff_0p6 = os.path.join(current_path, 'SpikestdCoact_output_1min_20minbuff_0p6').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikestdCoact_output_1min_20minbuff_0p6):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikestdCoact_output_1min_20minbuff_0p6, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_Normal[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_Normal[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_Normal[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_Normal[count].set_xlim(time[0]/3600,EndBaseline_Normal[count]/3600) #limiting to baseline data only
                # #ax_Normal[count].axvline(x=EndBaseline_Normal[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_Normal[count].tick_params(axis="x", labelsize=3)
                # ax_Normal[count].set_yticks([0,1])    
                # ax_Normal[count].spines["top"].set_visible(False)  
                # ax_Normal[count].spines["right"].set_visible(False)  
                # ax_Normal[count].spines["bottom"].set_visible(False)  
                # ax_Normal[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikestdCoact_output_1min_20minbuff_0p6" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikestdCoact_output_1min_20minbuff_0p6" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikestdCoact_output_1min_20minbuff_0p6"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_Normal_SpikestdCoact_output_1min_20minbuff_0p6" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 
     
# In[ ]: SpikestdCoact_output_1min_20minbuff_0p75

#threshold = 50
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_Normal = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_Normal = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_Normal_state_title= "States for Normal animals from SpikestdCoact_output_1min_20minbuff_0p75, threshold = " + str(threshold)
# fig.suptitle(str_Normal_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(Normal_path, filename)
    current_path_SpikestdCoact_output_1min_20minbuff_0p75 = os.path.join(current_path, 'SpikestdCoact_output_1min_20minbuff_0p75').replace("\\","/")

    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikestdCoact_output_1min_20minbuff_0p75):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikestdCoact_output_1min_20minbuff_0p75, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_Normal[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_Normal[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_Normal[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_Normal[count].set_xlim(time[0]/3600,EndBaseline_Normal[count]/3600) #limiting to baseline data only
                # #ax_Normal[count].axvline(x=EndBaseline_Normal[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_Normal[count].tick_params(axis="x", labelsize=3)
                # ax_Normal[count].set_yticks([0,1])    
                # ax_Normal[count].spines["top"].set_visible(False)  
                # ax_Normal[count].spines["right"].set_visible(False)  
                # ax_Normal[count].spines["bottom"].set_visible(False)  
                # ax_Normal[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikestdCoact_output_1min_20minbuff_0p75" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikestdCoact_output_1min_20minbuff_0p75" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikestdCoact_output_1min_20minbuff_0p75"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_Normal_SpikestdCoact_output_1min_20minbuff_0p75" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 

# In[ ]: SpikestdCoact_output_1min_20minbuff_0p9 

#threshold = 50
animals = list()
coactivity_stats_filepaths = list()
state_timestamp_Normal = []
bsstats_all = list()
split_char_animal="pig"
num_bs_replicates=1000 #Change to 50000

# fig, ax_Normal = plt.subplots(figsize = (22,12), nrows = len(filenames), ncols = 1)
# str_Normal_state_title= "States for Normal animals from SpikestdCoact_output_1min_20minbuff_0p9, threshold = " + str(threshold)
# fig.suptitle(str_Normal_state_title, fontsize=16)

count = 0
for filename in filenames:

    current_path = os.path.join(Normal_path, filename)
    current_path_SpikestdCoact_output_1min_20minbuff_0p9 = os.path.join(current_path, 'SpikestdCoact_output_1min_20minbuff_0p9').replace("\\","/")
    
    current_animal = filename.split(split_char_animal)[1]
    animals.append(filename)
    
    
    for root, dirs, files in os.walk(current_path_SpikestdCoact_output_1min_20minbuff_0p9):
        #print(files)
       
        for name in files:
            
            if name.startswith(("coactivity_stats.csv")):

                coactivity_stats_filepath = os.path.join(current_path_SpikestdCoact_output_1min_20minbuff_0p9, name).replace("\\","/") ## FOR WINDOWS BACKSLASH
                coactivity_stats_filepaths.append(coactivity_stats_filepath)
                
                str_current = "current path = " + coactivity_stats_filepath
                print(str_current)
                
                df = pd.read_csv(coactivity_stats_filepath)
                time = df['time']
                stats = df['coactivity_stat']
                
                #limit time before end of baseline
                index = time < EndBaseline_Normal[count]
                time = time[index]
                stats = stats[index]
                
                #convert to lists
                time = time.tolist()
                stats = stats.tolist()
                               
                ########################### UNCOMMENT TO PLOT STATES (COMMENT THE FIGURES BELOW FIRST) ###################################################### 
                # fill figure
                # ax_Normal[count].plot(time/3600, state_array ,'--', color = 'orchid', alpha=0.8)
                # ax_Normal[count].set_xticks(np.array(transition_timestamp)/3600)
                # ax_Normal[count].set_xlim(time[0]/3600,EndBaseline_Normal[count]/3600) #limiting to baseline data only
                # #ax_Normal[count].axvline(x=EndBaseline_Normal[count]/3600, color = 'black', linewidth = lw_EndBaseline) #full exp, mark end of baseline for each
                # ax_Normal[count].tick_params(axis="x", labelsize=3)
                # ax_Normal[count].set_yticks([0,1])    
                # ax_Normal[count].spines["top"].set_visible(False)  
                # ax_Normal[count].spines["right"].set_visible(False)  
                # ax_Normal[count].spines["bottom"].set_visible(False)  
                # ax_Normal[count].set_ylabel((''.join(filter(lambda i: i.isdigit(), current_animal))), fontsize=12)
                count = count + 1
                
                ########################## BOOTSTRAP (not sure what to bootstrap, was event rate definition len(events)/duration?) #################################
                
                #values = np.diff(np.array(transition_timestamp)) #data to bootstrap
                
                if len(stats) > 0:
                    # Draw N bootstrap replicates
                    denom = time[-1] - time[0] #hard coded
                    bs_replicates_values = draw_bs_replicates(denom,time, stats, num_bs_replicates)
                    
    
                    ########################### COMMENT IF PLOTTING STATES  ###################################################### 
    
                    # Plot the PDF for bootstrap replicates as histogram & save fig
                    plt.hist(bs_replicates_values,bins=30)
                    
                    lower=5
                    upper=95
                    # Showing the related percentiles
                    plt.axvline(x=np.percentile(bs_replicates_values,[lower]), ymin=0, ymax=1,label='5th percentile',c='y')
                    plt.axvline(x=np.percentile(bs_replicates_values,[upper]), ymin=0, ymax=1,label='95th percentile',c='r')
                    
                    plt.xlabel("Event rate")
                    plt.ylabel("Probability Density Function")
                    plt.title("pig" + current_animal + " SpikestdCoact_output_1min_20minbuff_0p9" +" Th: " + str(threshold))
                    plt.legend()
                    str_PDF_savefig_pdf= "pig" + str(current_animal) + "_PDF_SpikestdCoact_output_1min_20minbuff_0p9" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.pdf"
                    plt.savefig(str_PDF_savefig_pdf)    
                    plt.show()
                    
                    # Get the bootstrapped stats
                    bs_mean = np.mean(bs_replicates_values)
                    bs_std = np.std(bs_replicates_values)
                    ci = np.percentile(bs_replicates_values,[lower,upper])
                    ci_width = np.diff(ci)

                    # Print stuff
                    #print("event rate replicates: ",bs_replicates_values)
                    print("pig" + str(current_animal)+ " bootstrapped mean: ",bs_mean)
                    print( "pig" + str(current_animal) + " bootstrapped std: ",bs_std)
                    print("pig" + str(current_animal) + " bootstrapped ci: ",ci)
                    print("pig" + str(current_animal) + " bootstrapped ci width: ",ci_width)
                    
                    bsstats_concat = np.concatenate((np.array([bs_mean]),np.array([bs_std]),ci,ci_width))  
                    bsstats_all.append(bsstats_concat)     
                    
                else:
                    print(current_animal + "has no transition timestamps for threshold = " + str(threshold))
                    bsstats_concat = [999,999,999,999,999]
                    bsstats_all.append(bsstats_concat)
                    
                
df_bsstats = pd.DataFrame(bsstats_all)   

#rename columns
df_bsstats.rename(columns = {0 :'mean', 1 :'std', 2 :'lower', 3 :'upper', 4 :'ci_width'}, inplace = True)        
df_bsstats['state_threshold'] = threshold
df_bsstats['exceedance'] = "SpikestdCoact_output_1min_20minbuff_0p9"
df_bsstats['animal'] = filenames

#reindex column titles

column_titles = ['animal', 'mean', 'std', 'lower','upper','ci_width','state_threshold','exceedance']

df_bsstats=df_bsstats.reindex(columns=column_titles)

# add details to title
str_csv = "bsstats_Normal_SpikestdCoact_output_1min_20minbuff_0p9" + "_Thr" +  str(threshold) + "_BS" + str(num_bs_replicates) + "EvRate_base.csv"                                              

# save csv
df_bsstats.to_csv(str_csv, index=False) 

# In[ ]: Plot obtained results from real data
    
"""
CASE: SpikerateCoact			
animal	min of exceedance & state threshold combination	winner exceedance	winner state_threshold
pig1666	0p9	70
pig1670	0p9	90
pig1690pvccmrtx	0p75	90
pig1692chronicPVCRTX	0p9	90
pig1767pvc	0p9	90
pig1768	0p9	90
pig1770	0p9	90
pig1774pvc	0p9	90
pig1841pvc	0p9	90
pig1843pvc	0p9	90
pig1844	0p9	90
pig1720	0p9	60
pig1721	0p9	90
pig1723	0p9	80
pig1740	0p9	90
pig1741	0p9	90
pig1742	0p9	90

CASE: SpikestdCoact			
animal	min of exceedance & state threshold combination	winner exceedance	winner state_threshold
pig1666	0p9	90
pig1670	0p9	90
pig1690pvccmrtx	0p9	90
pig1692chronicPVCRTX	0p9	90
pig1767pvc	0p9	60
pig1768	0p9	90
pig1770	0p9	90
pig1774pvc	0p9	90
pig1841pvc	0p9	90
pig1843pvc	0p9	90
pig1844	0p9	90
pig1720	0p9	90
pig1721	0p9	90
pig1723	0p9	90
pig1740	0p9	90
pig1741	0p9	90
pig1742	0p9	90
        
"""