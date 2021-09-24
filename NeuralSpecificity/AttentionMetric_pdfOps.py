#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import sys
import datetime
from PyPDF2 import PdfFileMerger

# In[]:

def mergePDF(
        animal_name="",
        filepath_df="",
        outdir="",
        diary_file="",
        probe_df={},
        event=""):

    # save default printing settings
    # write process to diary file
    default_output = sys.stdout
    diary = open(diary_file, "a+")
    sys.stdout = diary
    
    # create filepath_df reverse lookup: name to index in filepath_df
    name_to_index = {k:v for v,k in enumerate(filepath_df["channel"])}

    for shank in probe_df.columns:

        # create merger instance
        merger = PdfFileMerger()
        
        # physically order shank
        physord={}
        for key, channel in enumerate(probe_df[shank]):
            physord[key+1] = channel
    
        # append files
        print('MERGE PDF: physically ordered merge ')
        for _, (_, ch_name_pos) in enumerate(physord.items()):
            try:
                print('MERGE PDF: found ',ch_name_pos)
                print(filepath_df.reset_index()["filename"][name_to_index[ch_name_pos]])
                merger.append(filepath_df.reset_index()["filename"][name_to_index[ch_name_pos]])
            except:
                print('MERGE PDF: not found ',ch_name_pos)
                
        # save merged file
        merger.write(outdir + "/" + animal_name + '_' + event + shank + '.pdf')
        
        # close merger instance
        merger.close()

    sys.stdout = default_output
    diary.close()

    return

# In[]:
# get dataFrame for all channels at once
def getFileDataframe(src_dir, event, diary_file, split_primary='n', split_backup='g'):

    default_output = sys.stdout
    diary = open(diary_file, "a+")
    sys.stdout = diary
 
    print('\ngetFileDataframe: NEW EVENT ', event)
    filepaths_with_event = []
    ch_names = np.array([]).astype(int)
    ch_names_with_event = []
    for ch_folder in sorted(os.listdir(src_dir)):
        print('getFileDataframe: Folder ',ch_folder)
        try:
            ch_num = int(ch_folder.split(split_primary)[-1])
            ch_names = np.append(ch_names, ch_num)
        except:
            ch_num = int(ch_folder.split(split_backup)[-1])
            ch_names = np.append(ch_names, ch_num)
            
        for file in os.listdir(src_dir + ch_folder):
            try:
                # find actual_event in the file
                actual_event = file.split('Attention_')[1].split('Metric')[0]
                # check for event equality: avoiding RS11 contains RS1
                # cannot use if event in file since RS11 contains RS1
                if event == actual_event:
                    filepaths_with_event.append(src_dir + ch_folder + "/" + file)
                    ch_names_with_event.append(ch_num)
                    print('getFileDataframe: Found: event, ch_num \n', event, ch_num)
            except:
                print('getFileDataFrame: not found event: \n',event, ch_num)

    # store sorted in dataframe
    df = pd.DataFrame({"filename": filepaths_with_event, "channel": ch_names_with_event})
    df = df.sort_values("channel")
    
    # print df
    print('getFileDataframe: channels with event', event)
    print(df)

    # reset file print pointer
    sys.stdout = default_output
    diary.close()

    return df

# In[]:
def run_pdfOps(
    animal_name,
    filepath_df,
    outdir,
    diary_file,
    probe_df,
    event
    ):

    os.nice(10)
    # save default printing settings
    # write process to diary file
    default_output = sys.stdout
    diary = open(diary_file, "a+")
    sys.stdout = diary

    mergePDF(
        animal_name=animal_name,
        filepath_df=filepath_df,
        outdir=outdir + "/",
        diary_file=diary_file,
        probe_df=probe_df,
        event=event
    )

    sys.stdout = default_output
    diary.close()

    return

# In[]:
if __name__ == "__main__":

    # Set the max number of processes to run concurrently
    # The machine does't like running more than 5
    maxProc = 4

    # special case name
    special_case = '_1min_20minbuff'
    
    # set primary channel split letter
    split_primary = 'l'
    split_backup = 'n'

    # Make sure directories DO NOT end in '/' !!!
    # example /media/dal/AWS/pigdata/pig1234/ml2_output is CORRECT!!!

    src_dirlist = [
                    "/media/dal/AWS/NormalAnimals/pig1720/AttentionMetric",
                    "/media/dal/AWS/NormalAnimals/pig1721/AttentionMetric",
                    "/media/dal/AWS/NormalAnimals/pig1723/AttentionMetric",
                    "/media/dal/AWS/NormalAnimals/pig1740/AttentionMetric",
                    "/media/dal/AWS/NormalAnimals/pig1741/AttentionMetric",
                    "/media/dal/AWS/NormalAnimals/pig1742/AttentionMetric",                   
                   ]
   
    # physical ordering: {location in figure:channel#}
    #physord = {1:9,2:10,3:11,4:12,5:13,6:14,7:15,8:16,9:1,10:2,11:3,12:4,13:5,14:6,15:7,16:8}
    #physord={}
    #for keys in np.arange(256):
    #    physord[keys+1]=keys+1
    
    # ANIMAL: list the comment files
    commentfile_list = [
                        '/media/dal/AWS/NormalAnimals/pig1720/CommentFiles/pig1720_comment_summary.csv',
                        '/media/dal/AWS/NormalAnimals/pig1721/CommentFiles/pig1721_comment_summary.csv',
                        '/media/dal/AWS/NormalAnimals/pig1723/CommentFiles/pig1723_comment_summary.csv',
                        '/media/dal/AWS/NormalAnimals/pig1740/CommentFiles/pig1740_comment_summary.csv',
                        '/media/dal/AWS/NormalAnimals/pig1741/CommentFiles/pig1741_comment_summary.csv',
                        '/media/dal/AWS/NormalAnimals/pig1742/CommentFiles/pig1742_comment_summary.csv',
                        ]
    
    # ANIMAL: probe file
    probefile_list = ['/media/dal/AWS/NormalAnimals/pig1720/ProbeFiles/lma16channel_probe.csv',
                      '/media/dal/AWS/NormalAnimals/pig1721/ProbeFiles/lma16channel_probe.csv',
                      '/media/dal/AWS/NormalAnimals/pig1723/ProbeFiles/lma16channel_probe.csv',
                      '/media/dal/AWS/NormalAnimals/pig1740/ProbeFiles/lma16channel_probe.csv',
                      '/media/dal/AWS/NormalAnimals/pig1741/ProbeFiles/lma16channel_probe.csv',
                      '/media/dal/AWS/NormalAnimals/pig1742/ProbeFiles/lma16channel_probe.csv',
                      ]
    
    
# In[]:    
    pd.options.display.max_colwidth = 200
    for src_dir, comment_file, probe_file in zip(src_dirlist, commentfile_list, probefile_list):
        
        print(f'src_dir {src_dir} comment_file {comment_file} probe_file {probe_file}')
        # add special case instance
        src_dir = src_dir + special_case

        # make the pdfOps_output directory
        f = src_dir.split("/")[-1:][0]
        pdfOps_outputfolder = src_dir.replace(f, "AttentionMetric_pdfOps" + special_case)
        
        # get animal name
        animal_name = "pig" + src_dir.split("pig")[1].split("/")[0]
        print('MAIN: animal name ',animal_name)

        try:
            outdir = pdfOps_outputfolder
            os.mkdir(outdir)
        except:
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m_%d_%H:%M:%S")
            outdir = pdfOps_outputfolder[:-1] + now
            os.mkdir(outdir)

        # set diary file
        diary_file = outdir + "/diary_Attention_pdfOps.txt"

        # create event names
        # build dataframe
        try:
            comment_df = pd.read_csv(comment_file)
            # create np(events_list)
            event_list = np.array(comment_df['events'].to_list())
            # create np(event labels)
            event_labels = np.array(comment_df['labels'].to_list())            
            # create event starts
            start_list = np.array(np.nonzero(event_list=='start')[0])
            # create event_names
            event_names = event_labels[start_list]
            # enumerate
            event_names = [event_names[num] + str(num) for num in np.arange(len(event_names))]
            # append AllEvents to get full expt too
            event_names = np.append(event_names, "AllEvents")
            # note: these two have an extra subscript: was used to make file
            # names more readable
            event_names = np.append(event_names, "AllEventsRandom_")
            event_names = np.append(event_names, "AllEventsSample_") 
           # notify
            print('MAIN: Comment File Found')            
            f = open(diary_file, "a+")
            f.write('MAIN: Comment File Found')
            f.close()
            
        except:
            # notify
            print('MAIN: No Comment File Found')
            f = open(diary_file, "a+")            
            f.write('MAIN: No Comment File Found \n')
            f.write(src_dir)
            f.close()


        # animal: probe df
        probe_df = pd.read_csv(probe_file)
        print('probe head ')
        print(probe_df.head())

        # animal: merge list of events
        for event in event_names:
            print('MAIN: event ',event)
            
            try:
                # adding trailing '/' to end of src_dir
                print('MAIN: src dir ',src_dir)
                filepath_df = getFileDataframe(src_dir + "/", 
                                               event, diary_file, 
                                               split_primary=split_primary, 
                                               split_backup=split_backup
                                               )
                print(filepath_df["filename"])
                
                # start a process of running AM
                if not pd.DataFrame(filepath_df).empty:
                    p = mp.Process(
                        target=run_pdfOps,
                        args=[
                            animal_name,
                            filepath_df,
                            outdir,
                            diary_file,
                            probe_df,
                            event
                        ],
                    )
            
                    p.start()
            
                    # dont start more than maxProc
                    while len(mp.active_children()) == maxProc:
                        continue                
            except:
                # notify
                print('\n***MAIN: Event Not Found***: ', event)
                f = open(diary_file, "a+")            
                f.write('\n***MAIN: Event Not Found*** ')
                f.write(event)
                f.write("\n")
                f.write(src_dir)
                f.close()
                

    
    # causes code to wait until done
    while len(mp.active_children()) > 0:
        continue

    print("MAIN: pdf Ops Done")
