#!/usr/bin/env python
# coding: utf-8

# In[]
import os
import pandas as pd
from SpikeRateCoactivityMC import SpikeRateCoact
import multiprocessing as mp
import sys
import datetime
import numpy as np

# In[]
def getFilepaths(NA_SpikeRateoutputs):

    filepaths = []
    for folder in sorted(os.listdir(NA_SpikeRateoutputs)):

        for file in os.listdir(NA_SpikeRateoutputs + folder):

            if "outputSpike_uniform_time_rate" in file:
                # if int(file[:-4].split('icn')[-1:][0]) in chan_select:
                filepaths.append(NA_SpikeRateoutputs + folder + "/" + file)

    return filepaths

# run SpikeRateCoactivityMC.SpikeRateCoact
def runSRCoact(
   animal_name,
   diary_file,
   filepaths,
   chList,
   ch_numbers,
   comment_df,
   outdir,
   window,
   ytick_precision,
   ytick_downsample,
   name,
   before_buffer,
   after_buffer,
   corr_threshold,
   ylim_min,
   ylim_max
):
    
    #os.nice(10)
    # save default printing settings
    # write process to diary file
    default_output = sys.stdout
    diary = open(diary_file, "a+")
    sys.stdout = diary

    print('performing ',name)
    print('outdir ',outdir)
    print('ytick_precision ',ytick_precision)
    print('ytick_downsample ',ytick_downsample)
    print('running spikerate coact')
    print('animal_name: ',animal_name)
    for i, path in enumerate(filepaths):
        print('filepaths: ',i, path)
    print('channel numbers: ',ch_numbers)

    SpikeRateCoact(
        animal_name=animal_name,
        filepaths=filepaths,
        ch_list=chList,
        ch_numbers=ch_numbers,
        comment_df=comment_df,
        outdir=outdir,
        window=window,
        ytick_precision=ytick_precision,
        ytick_downsample=ytick_downsample,
        name=name,
        before_buffer=before_buffer,
        after_buffer=after_buffer,
        corr_threshold=corr_threshold,
        ylim_min=ylim_min,
        ylim_max=ylim_max
 )

    sys.stdout = default_output
    diary.close()

    return


if __name__ == "__main__":

    
# In[]:
# ANIMAL DATA

    # global coactivity parameters
    # set window: centered: odd length: time = window * dt (see SpikeRateCoactivityMC.py)
    # default is dt = 0.02 (1 min = 3001, 15 min = 45000)
    window = 501 * 5
    # plot parameter
    ytick_downsample=2
    # plot parameter
    ytick_precision=0

    
    # Set the max number of processes to run concurrently
    # The machine does't like running more than 5
    maxProc = 4


    # set before event buffer (secs) ... use 300.0 for 1min window above
    before_buffer = 360.0
    # set after event buffer (secs)
    after_buffer = 360.0

    #
    # DO NOT FORGET TO SET THE CHANNEL SPLIT LETTER
    # set primary channel split letter
    split_primary = 'n' #JOB - Fix dependecy on specific filenames 
    split_backup = 'g'
    
    #
    # set directory to special case if needed or special_case=''
    special_case = '_10s_6minbuff_corr_thr_0p9'
    
    #
    # corr_threshold for coactivity_stats
    corr_threshold = 0.6
    
    #
    # coactivity_stats y limits:  percentage var
    ylim_min = 0.0
    ylim_max = 100.0

    #   
    # DO NOT FORGET TO SET WHAT JOB IS BEING PERFORMED
    # set what want to do rate='rate' and std='std'
    name = 'rate'
   
    # Make sure directories DO NOT end in '/'
    # ANIMAL: example /media/dal/AWS/pigdata/pig1234/ml2_output and NO '/' at end

    # Make sure directories DO NOT end in '/'
    # ANIMAL: example /media/dal/AWS/pigdata/pig1234/ml2_output and NO '/' at end
    NA_dirlist = ["../CoactivityCofluctuationEventRate/Animal_Data/NeuralOutSpike"] #Directry containing  channel folders having outspikedata
    
    # ANIMAL: list the comment files
    commentfile_list = ['../CoactivityCofluctuationEventRate/Animal_Data/CommentFiles']

    # ANIMAL: ch_numbers: physically ordered list
    #ch_numbers = np.array([9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8]).astype(int)
    ch_numbers = np.array([1, 2, 3, 5]).astype(int)


    # let filenames completely print
    pd.options.display.max_colwidth = 200
    
    # for ANIMALS
    for NA_outputdir, comment_file in zip(NA_dirlist, commentfile_list):
        
        # ANIMAL name
        animal_name = 'Animal' #NA_outputdir.split('Animals/')[1].split('/')[0]
           
        # make comment file dataframe
        try:
            comment_df = pd.read_csv(comment_file)
            print('Found Comment File ',comment_file)
        except:
            #create wrong type - will exit the try in plot routine later
            comment_df = ""
            print('No Comment File Found')

        # make the output directory
        # detect stuff past last '/'
        f = NA_outputdir.split("/")[-1:][0]
        print(f)
        # replace NeuralAutocorrSingle_Output with SpikRateCoact_Output
        outputfolder = NA_outputdir.replace(f, "Spike"+name+"Coact_output"+special_case)
        print(outputfolder)
        try:
            outdir = outputfolder
            os.mkdir(outdir)
        except:
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m_%d_%H:%M:%S")
            outdir = outputfolder[:-1] + now
            os.mkdir(outdir)
        print("os list ")
        print(os.listdir(NA_outputdir))

        # make list of channel number
        chList = np.array([]).astype(int)
        for ch in os.listdir(NA_outputdir):
            try:
                chList = np.append(chList, int(ch.split('_')[-1][-1]))
            except:
                print('Exception thrown : channel number not found')
                #chList = np.append(chList, int(ch.split(split_backup)[-1]))
        print('Channel list', chList)
        # get all available channel Filepaths of SpikeRate
        filepaths = getFilepaths(NA_outputdir + "/")
        print('File paths', filepaths)

        # output file for SpikeRateCoact diary
        diary_file = outputfolder + "/diarySpike" +name + "Coact.txt"
        print('diary_file', diary_file)


        # ANIMAL: SpikeRateCoact compute across events for an ANIMAL
        p = mp.Process(
            target=runSRCoact,
            args=[
                    animal_name,
                    diary_file,
                    filepaths,
                    chList,
                    ch_numbers,
                    comment_df,
                    outdir,
                    window,
                    ytick_precision,
                    ytick_downsample,
                    name,
                    before_buffer,
                    after_buffer,
                    corr_threshold,
                    ylim_min,
                    ylim_max
           ],
        )

        # start process
        p.start()

        # dont start more than maxProc
        while len(mp.active_children()) == maxProc:
            continue

    # causes code to wait until done
    while len(mp.active_children()) > 0:
        continue

    print("Spike"+name+"Coact Done")
