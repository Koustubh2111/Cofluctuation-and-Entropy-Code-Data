#!/usr/bin/env python
# coding: utf-8

# In[]
import os
import pandas as pd
from AttentionMetricMC import AttentionMetric
import multiprocessing as mp
import sys
import datetime
import numpy as np

# In[]
def getFilepaths(ml2_outputs):

    filepaths = []
    print('Run through channel folders in ML2 folder of an animal')
    for folder in sorted(os.listdir(ml2_outputs)):

        for file in os.listdir(ml2_outputs + folder):

            if "Uncurated" in file:
                # if int(file[:-4].split('icn')[-1:][0]) in chan_select:
                filepaths.append(ml2_outputs + folder + "/" + file)

    return filepaths


def runPigAM(
    channel_name,
    filepath_df,
    outdir,
    diary_file,
    metadataAM_file,
    hasLVP,
    hasRESP,
    target_file,
    comment_df,
    badtar_lower_lim,
    badtar_upper_lim,
    hard_threshold_rel_to_mean,
    att_metric,
    before_buffer,
    after_buffer
):

    #os.nice(10)
    # save default printing settings
    # write process to diary file
    default_output = sys.stdout
    diary = open(diary_file, "a+")
    sys.stdout = diary

    AttentionMetric(
        channel_name=channel_name,
        file_df=filepath_df,
        target_file=target_file,
        comment_df=comment_df,
        AM_metadatafile=metadataAM_file,
        outdir=outdir + "/",
        hasLVP=hasLVP,
        hasRESP=hasRESP,
        uncurated=True,
        badtar_lower_lim=badtar_lower_lim,
        badtar_upper_lim=badtar_upper_lim,
        hard_threshold_rel_to_mean=hard_threshold_rel_to_mean,
        att_metric=att_metric,
        before_buffer=before_buffer,
        after_buffer=after_buffer
 )

    sys.stdout = default_output
    diary.close()

    return


if __name__ == "__main__":

    # Set the max number of processes to run concurrently
    # The machine doesn't like running more than 5

    maxProc = 4
    
    # special case name
    special_case = '_20s_6minbuff'
    
    # special case ML2
    #special_case_ML2_output = ''
   
    #
    # DO NOT FORGET TO SET THE CHANNEL SPLIT LETTER
    # set primary channel split letter
    split_primary = 'l'
    split_backup = 'n'
    
    # set before event buffer (secs)
    # use 300 sec buffer for 60 sec window
    before_buffer = 360.0
    # set after event buffer (secs)
    after_buffer = 360.0

    # Make sure directories DO NOT end in '/'

    #ML2 output directory without the '/' for traversal
    ml2_dirlist = ['../Animals/Animal_Data/Animal1/ML2_Output',\
                  '../Animals/Animal_Data/Animal2/ML2_Output']                  

    # ANIMAL: List the target files: CHOOSE ONE TARGET
    # Contans 2mins of entropy
    targetfile_list = [
                    '../Animals/Animal_Data/Animal1/TargetFiles/A1_lvp.mat',\
                    '../Animals/Animal_Data/Animal2/TargetFiles/A2_lvp.mat']

    # ANIMAL: list the comment files
    #Contains the comment files to include the events used in the animal experiment  
    commentfile_list = [
                        '../Animals/Animal_Data/Animal1/CommentFiles/Animal_comment_summary.csv',\
                        '../Animals/Animal_Data/Animal2/CommentFiles/Animal_comment_summary.csv']

    # lvp: bad target lower and upper limits  
    badtarlower_list = [
                        -np.inf, 
                        -np.inf, 
                        -np.inf, 
                        -10.0, 
                        -np.inf, 
                        -np.inf, 
                        ]
    badtarupper_list = [
                        np.inf,
                        np.inf,
                        np.inf,
                        205.0,
                        np.inf,
                        np.inf,
                        ]



    # if different for animals then could change to list
    hasLVP = True
    hasRESP = False

    # Decide on hard thresh and att metric
    hard_threshold_rel_to_mean = False
    att_metric = True

    pd.options.display.max_colwidth = 200

    for ml2_outputdir, target_file, comment_file, badtar_lower_lim, badtar_upper_lim in \
        zip(ml2_dirlist, targetfile_list, commentfile_list, badtarlower_list, badtarupper_list):
        
        #ml2_outputdir  - one animal's channel ML2 output folders
        #target_file - the .mat file of LVP of the same animal
        #comment_file - thecomments of the animal
    
        # special case ML_output
	#ECG or RESP cases; different cases output names
        #ml2_outputdir = ml2_outputdir + special_case_ML2_output
           
        # make comment file dataframe
        try:
            #READ THE COMMENT FILE
            comment_df = pd.read_csv(comment_file)
            print('Found Comment File ',comment_file)
        except:
            #create wrong type - will exit the try in plot routine later
            comment_df = ""
            print('No Comment File Found')

        # make the output directory
        # detect stuff past last '/'
        f = ml2_outputdir.split("/")[-1:][0] #The 'ML2_output' string of the animal
        print(f)
        # replacement done if using standard dir/channel/file naming

        # replace ML2_Output with AttentionMetric
        replace_str = "NeuralSpecificity" + special_case
        am_outputfolder = ml2_outputdir.replace(f, replace_str)

        #am_outputfolder =   f +\
        #                    ml2_outputdir.split('_')[1] +\
        #                    "/" +\
        #                    "AttentionMetric" +\
        #                    #special_case_ML2_output +\
        #                    special_case
        print(am_outputfolder)
        try:
            outdir = am_outputfolder
            os.mkdir(outdir)
        except:
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m_%d_%H:%M:%S")
            outdir = am_outputfolder[:-1] + now
            os.mkdir(outdir)
        print("os list ")
        print(os.listdir(ml2_outputdir))

        # make list of channel number
        chList = np.array([]).astype(int)
        testfilepaths = []

        for folder in sorted(os.listdir(ml2_outputdir)):
            #print('folder ',folder)
            for file in os.listdir(ml2_outputdir + "/" + folder):
                #print('folder file ',folder, file)
                if "Uncurated" in file:
                    # if int(file[:-4].split('icn')[-1:][0]) in chan_select:
                    testfilepaths.append(ml2_outputdir + "/" + folder + "/" + file)                            
                    try: 
                        chList = np.append(chList, int(folder.split('_')[-1][-1])) #Channel_number as integer
                    except:
                        print('Exception thrown: Channel number invalid')
                        #chList = np.append(chList, int(folder.split(split_backup)[-1]))

        # uncomment for debugging
        #print(testfilepaths)
        # chList has SAME order as os.listdir(ml2_outputdir)
        for (ch, chN) in zip(os.listdir(ml2_outputdir), chList):

            # get all Filepaths of uncurated (default=uncurated)
            #Returns uncurated csvs of all channels of an animal
            filepaths = getFilepaths(ml2_outputdir + "/")

            # set channel output directory
            # add channel=ch to e.g. outdir_ch = outdir + '/' + [ch = e.g. icn1]
            outdir_ch = outdir + "/" + ch
            print(outdir_ch)

            # make channel output directory
            try:
                os.mkdir(outdir_ch)
            except:
                now = datetime.datetime.now()
                now = now.strftime("%Y-%m_%d_%H:%M:%S")
                outdir_ch = outdir_ch + now
                os.mkdir(outdir_ch)


            # init all channels to -9 = not used
            chListUsage = np.ones(len(filepaths)).astype(int) * -9
            # replace desired channel with its number (required in AttMetric)
            chListUsage[np.nonzero(chList == chN)[0][0]] = chN

            # file directories for AM
            diary_file = outdir_ch + "/diaryAM" + ch + ".txt"
            metadataAM_file = outdir_ch + "/metadata_AttentionMetric" + ch + ".txt"
            filepath_df = pd.DataFrame({"filename": filepaths, "channel": chListUsage})
            print(filepath_df)
            # start a process of running AM

            # diary dump
            # save default printing settings
            # write process to diary file
            default_output = sys.stdout
            diary = open(diary_file, "a+")
            sys.stdout = diary
 
            print('outdir ',outdir)
            print('before_buffer ',before_buffer)
            print('after buffer ',after_buffer)
            print('event split primary letter ',split_primary)
            print('event split backup letter ',split_backup)
            
            for i, [path, usage] in enumerate(zip(filepaths, chListUsage)):
                print('filepaths: ',i, path, usage)
            print('channel list: ',chList)
            
            sys.stdout = default_output
            diary.close()
            
            p = mp.Process(
                target=runPigAM,
                args=[
                    ch,
                    filepath_df,
                    outdir_ch,
                    diary_file,
                    metadataAM_file,
                    hasLVP,
                    hasRESP,
                    target_file,
                    comment_df,
                    badtar_lower_lim,
                    badtar_upper_lim,
                    hard_threshold_rel_to_mean,
                    att_metric,
                    before_buffer,
                    after_buffer
               ],
            )

            p.start()

            # dont start more than maxProc
            while len(mp.active_children()) == maxProc:
                continue

        # causes code to wait until done
        while len(mp.active_children()) > 0:
            continue

        print("Attention Metric Done")
