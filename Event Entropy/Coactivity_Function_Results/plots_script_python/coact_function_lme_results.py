# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:23:27 2021

@author: NGurel


############ MATLAB LINES: coact_function_fitlme.m #######################################
% 1. is entropy of event vs non event different regardless of animal type?
% Entropy_Mean ~ Event_type + (Coact_Type) + (1|Animal) + (1|Channel) : YES
% Entropy_Mean ~ Event_type + (Coact_Type) + (1|Animal) +
% (1|Baseline_Entropy_Mean) + (1|Channel) : YES
% Entropy_Std ~ Event_type + (Coact_Type) + (1|Animal) + (1|Channel) : YES
% Entropy_Std ~ Event_type + (Coact_Type) + (1|Animal) +
% (1|Baseline_Entropy_Std) + (1|Channel) : YES

lme=fitlme(T, 'Entropy_Mean ~ Event_type + Coact_Type + (1|Animal) + (1|Channel)') % p = 0.021851
lme=fitlme(T, 'Entropy_Mean ~ Event_type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel)') % p = 0.00061659 
lme=fitlme(T, ' Entropy_Std ~ Event_type + Coact_Type + (1|Animal) + (1|Channel)') % p =1.5192e-14
lme=fitlme(T, 'Entropy_Std ~ Event_type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Std) + (1|Channel)') % p = 3.9889e-17

% 2. is entropy of event vs non event different between animal types?
% Entropy_Mean ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) +
% (1|Channel) : NO
% Entropy_Mean ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel): NO
% Entropy_Std ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) +
% (1|Channel): YES
% Entropy_Std ~ Event_type + Animal_Type + (Coact_Type) + (1|Animal) +
% (1|Baseline_Entropy_Std) + (1|Channel): YES

lme=fitlme(T, 'Entropy_Mean ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Channel)') % % p(Event_type) = 0.021848,  p(Animal_Type) = 0.072525 
lme=fitlme(T, 'Entropy_Mean ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel)') % p(Event_type) = 0.00061668 , p(Animal_Type) = 0.075029
lme=fitlme(T, 'Entropy_Std ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Channel)') % p(Event_type) = 1.5252e-14  , p(Animal_Type)= 0.012124 
lme=fitlme(T, 'Entropy_Std ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Std) + (1|Channel)') % p(Event_type) =  4.0063e-17, p(Animal_Type) =  0.011961


############ FOLDER & FILE #######################################

FOLDER: C:\Users\ngurel\Documents\Stellate_Recording_Files\Coactivity_Function\coact_function_0825_csvs
FILE: coact_function_dataset_numerical.csv (ran in matlab)
PLOT FILE: coact_function_dataset_originalnames.csv
LME RESULTS: C:\Users\ngurel\Documents\Stellate_Recording_Files\Coactivity_Function\coact_function_0825_csvs\lme_results_v2_Fcoacttype
"""
# In[ ]: 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# In[ ]: ER filename
filename ='C:/Users/ngurel/Documents/Stellate_Recording_Files/Coactivity_Function/coact_function_0825_csvs/coact_function_dataset_originalnames.csv'

df = pd.read_csv(filename)

# In[ ]: lme=fitlme(T, 'Entropy_Mean ~ Event_type + Coact_Type + (1|Animal) + (1|Channel)') % p = 0.021851
pval = 0.021851 
fig_title = 'LME Pval = ' + str(pval)

colors_list = ['dodgerblue', 'magenta']

plt.figure(figsize=(4,4))

ax = sns.violinplot(x=df['Event_type'], y=df['Entropy_Mean'], palette = colors_list, linewidth = 2, scale='width', color="k" )
plt.setp(ax.collections, alpha=.8)
ax = sns.swarmplot(x=df['Event_type'], y=df['Entropy_Mean'], color="dimgray", size = 3, marker = 'v', alpha = 0.5)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 9)
# ax.set_xticklabels(['Normal', 'HF'], fontsize = 12)
# ax.set_xlabel('')
# ax.set_ylabel('')
ax.set_title(fig_title)
plt.savefig('entropymean_Feventtype_Fcoacttype_Ranimal_RChannel.pdf')

# In[ ]: ' Entropy_Std ~ Event_type + Coact_Type + (1|Animal) + (1|Channel)') % p =1.5192e-14

pval = 1.5192e-14 
fig_title = 'LME Pval = ' + str(pval)

colors_list = ['royalblue', 'orchid']

plt.figure(figsize=(5,4))

ax = sns.violinplot(x=df['Event_type'], y=df['Entropy_Std'], palette = colors_list, linewidth = 2, scale='width', color="k" )
plt.setp(ax.collections, alpha=.8)
ax = sns.swarmplot(x=df['Event_type'], y=df['Entropy_Std'], color="dimgray", size = 3, marker = 'v', alpha = 0.5)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 9)
# ax.set_xticklabels(['Normal', 'HF'], fontsize = 12)
# ax.set_xlabel('')
# ax.set_ylabel('')
ax.set_title(fig_title)
plt.savefig('entropystd_Feventtype_Fcoacttype_Ranimal_RChannel.pdf')

# In[ ]: 'Entropy_Mean ~ Event_type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel)') % p = 0.00061659 
pval = 0.00061659 
fig_title = 'LME Pval = ' + str(pval)

colors_list = ['dodgerblue', 'magenta']

plt.figure(figsize=(4,4))

ax = sns.violinplot(x=df['Event_type'], y=df['Entropy_Mean'], palette = colors_list, linewidth = 2, scale='width', color="k" )
plt.setp(ax.collections, alpha=.8)
ax = sns.swarmplot(x=df['Event_type'], y=df['Entropy_Mean'], color="dimgray", size = 3, marker = 'v', alpha = 0.5)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 9)
# ax.set_xticklabels(['Normal', 'HF'], fontsize = 12)
# ax.set_xlabel('')
# ax.set_ylabel('')
ax.set_title(fig_title)
plt.savefig('entropymean_Feventtype_Fcoacttype_Ranimal_RBaseEntMean_RChannel.pdf')

# In[ ]: 'Entropy_Std ~ Event_type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Std) + (1|Channel)') % p = 3.9889e-17

pval = 3.9889e-17
fig_title = 'LME Pval = ' + str(pval)

colors_list = ['royalblue', 'orchid']

plt.figure(figsize=(5,4))

ax = sns.violinplot(x=df['Event_type'], y=df['Entropy_Std'], palette = colors_list, linewidth = 2, scale='width', color="k" )
plt.setp(ax.collections, alpha=.8)
ax = sns.swarmplot(x=df['Event_type'], y=df['Entropy_Std'], color="dimgray", size = 3, marker = 'v', alpha = 0.5)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 9)
# ax.set_xticklabels(['Normal', 'HF'], fontsize = 12)
# ax.set_xlabel('')
# ax.set_ylabel('')
ax.set_title(fig_title)
plt.savefig('entropystd_Feventtype_Fcoacttype_Ranimal_RBaseEntStd_RChannel.pdf')

# In[ ]: 'Entropy_Mean ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Channel)') % % p(Event_type) = 0.021848,  p(Animal_Type) = 0.072525 
pval = 0.072525 
fig_title = 'LME Pval = ' + str(pval)

colors_list = ['dodgerblue', 'magenta']

plt.figure(figsize=(4,4))

ax = sns.violinplot(x=df['Animal_Type'], y=df['Entropy_Mean'], palette = colors_list, linewidth = 2, scale='width', color="k" )
plt.setp(ax.collections, alpha=.8)
ax = sns.swarmplot(x=df['Animal_Type'], y=df['Entropy_Mean'], color="dimgray", size = 3, marker = 'v', alpha = 0.5)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 9)
# ax.set_xticklabels(['Normal', 'HF'], fontsize = 12)
# ax.set_xlabel('')
# ax.set_ylabel('')
ax.set_title(fig_title)
plt.savefig('entropymean_Feventtype_Fanimaltype_Fcoacttype_Ranimal_RChannel.pdf')


# In[ ]: 'Entropy_Std ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Channel)') % p(Event_type) = 1.5252e-14  , p(Animal_Type)= 0.012124 

pval = 0.012124 
fig_title = 'LME Pval = ' + str(pval)

colors_list = ['royalblue', 'orchid']

plt.figure(figsize=(5,4))

ax = sns.violinplot(x=df['Animal_Type'], y=df['Entropy_Std'], palette = colors_list, linewidth = 2, scale='width', color="k" )
plt.setp(ax.collections, alpha=.8)
ax = sns.swarmplot(x=df['Animal_Type'], y=df['Entropy_Std'], color="dimgray", size = 3, marker = 'v', alpha = 0.5)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 9)
# ax.set_xticklabels(['Normal', 'HF'], fontsize = 12)
# ax.set_xlabel('')
# ax.set_ylabel('')
ax.set_title(fig_title)
plt.savefig('entropystd_Feventtype_Fanimaltype_Fcoacttype_Ranimal_RChannel.pdf')

# In[ ]: 'Entropy_Mean ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Mean) + (1|Channel)') % p(Event_type) = 0.00061668 , p(Animal_Type) = 0.075029
pval = 0.075029 
fig_title = 'LME Pval = ' + str(pval)

colors_list = ['dodgerblue', 'magenta']

plt.figure(figsize=(4,4))

ax = sns.violinplot(x=df['Animal_Type'], y=df['Entropy_Mean'], palette = colors_list, linewidth = 2, scale='width', color="k" )
plt.setp(ax.collections, alpha=.8)
ax = sns.swarmplot(x=df['Animal_Type'], y=df['Entropy_Mean'], color="dimgray", size = 3, marker = 'v', alpha = 0.5)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 9)
# ax.set_xticklabels(['Normal', 'HF'], fontsize = 12)
# ax.set_xlabel('')
# ax.set_ylabel('')
ax.set_title(fig_title)
plt.savefig('entropymean_Feventtype_Fanimaltype_Fcoacttype_Ranimal_RBaseEntMean_RChannel.pdf')

# In[ ]: 'Entropy_Std ~ Event_type + Animal_Type + Coact_Type + (1|Animal) + (1|Baseline_Entropy_Std) + (1|Channel)') % p(Event_type) =  4.0063e-17, p(Animal_Type) =  0.011961

pval =0.011961
fig_title = 'LME Pval = ' + str(pval)

colors_list = ['royalblue', 'orchid']

plt.figure(figsize=(5,4))

ax = sns.violinplot(x=df['Animal_Type'], y=df['Entropy_Std'], palette = colors_list, linewidth = 2, scale='width', color="k" )
plt.setp(ax.collections, alpha=.8)
ax = sns.swarmplot(x=df['Animal_Type'], y=df['Entropy_Std'], color="dimgray", size = 3, marker = 'v', alpha = 0.5)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 9)
# ax.set_xticklabels(['Normal', 'HF'], fontsize = 12)
# ax.set_xlabel('')
# ax.set_ylabel('')
ax.set_title(fig_title)
plt.savefig('entropystd_Feventtype_Fanimaltype_Fcoacttype_Ranimal_RBaseEntStd_RChannel.pdf')


















