#!/usr/bin/env python
# coding: utf-8

"""
@author: ackar

Future edits:
    - Could add argparse to edits params of ML2
        depends on how we want to do it though
    
"""

import os
from MultiLevel2MC import MultiLevel2
import sys
from multiprocessing import Process
import time
import datetime
import numpy as np
import h5py
from scipy import signal
import multiprocessing

# ourSmoothData imported to smooth resp & lvp data
def ourSmoothData(values, halfwin) :
    window = 2 * halfwin + 1
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    sma = np.insert(sma,0,values[0:halfwin])
    sma = np.append(sma,values[-halfwin:])
    return sma

####################################################
# function used to read in resp & lvp and smooth them
# if resp_file and lvp_file aren't passed in they're set to '' and all variables
# related to resp/lvp are set to defaults
# all data related is passed as a list [] with variables that're read in MultiLevel2MC
    
def read_LvpResp_data(resp_file = '',lvp_file = '') :
    
    try:
        
        # if the variable was passed, try doing this
        if lvp_file != '':
            # load
            L = h5py.File(lvp_file,'r')
            
            # abstract list
            L_list = [this_key for this_key in L ]
            
            # extract lvp
            # lvp values: : flatten for later cat
            lvpraw = np.array(L.get(L_list[0])['values']).flatten()
            
            # start time
            lvp_start = L.get(L_list[0])['start'][0][0]
            
            # lvp interval
            lvp_interval = L.get(L_list[0])['interval'][0][0]
            
            # close input file
            L.close()        
            
            lvpsmooth_local_max_win = 20
            lvpsmooth_local_min_win = 10
        
            lvpsmooth = ourSmoothData(lvpraw, lvpsmooth_local_max_win)
            
            for win in np.arange(lvpsmooth_local_max_win-1,lvpsmooth_local_min_win,-1):
                lvpsmooth = ourSmoothData(lvpsmooth, win)
                
            # make diff lvpmooth: remove effect of diff: smooth
            diff_lvpsmooth = np.diff(lvpsmooth)
            # add the first value so lengths are the same
            diff_lvpsmooth = np.concatenate([diff_lvpsmooth[0]*np.ones(1),diff_lvpsmooth ])
            # smooth 
            for x in range(4,0,-1): diff_lvpsmooth = ourSmoothData(diff_lvpsmooth,x)
            
            # lvp raw  isnt useanywhere ourside of here so to save MEM sending '1'
            lvp_data = [lvpraw,lvpsmooth,diff_lvpsmooth,lvp_start,lvp_interval]
            
        else:
            # no data, set defaults
            lvp_data = [np.zeros(1),np.zeros(1),np.zeros(1),-9,-9] 
            
    except Exception as e:
        print('Error Raised loading lvp data\n\n')
        print(e)  
        sys.quit()
            
    try:
        
        if resp_file != '':

            R = h5py.File(resp_file,'r')
            
            #abstract list
            R_list = list([])
            for this_key in R :
                R_list.append(this_key)
        
            # extract respiratory
            # valRes: : flatten for later cat
            # not called raw because raw is never used 
            respraw = np.array(R.get(R_list[0])['values']).flatten()
        
            # start time
            resp_start = R.get(R_list[0])['start'][0][0]
            
            # resp interval
            resp_interval = R.get(R_list[0])['interval'][0][0]
            
            # close input file
            R.close()        
            resp = respraw
            for x in range(18,14,-1): resp = ourSmoothData(resp,x)
            # detrend the data
            resp = signal.detrend(resp)  
            for x in range(85,81,-1): resp = ourSmoothData(resp,x)
        
        
            resp_data = [respraw,resp,resp_start,resp_interval]
            
        else:
            
            resp_data = [np.array([]),np.ones(1),-9,-9]
            
    except Exception as e:
        
        print('Error raised Loading Resp Data\n\n\n')
        print(e)
        sys.quit()

        resp_data = [np.array([]),np.zeros(1),-9,-9] 
    
    return lvp_data, resp_data



def runChannelML2(resp_data, lvp_data, metadata1_file, mat_file = '',outputspike_file = '',  diary_file = '', metdata2_file = '', gui_file = '',\
                      spikefortemplate_file = '', uncurated_file = '', curated_file = '' ):
    
    print('Call runChannelML2')
    # Set all the printing outputs to a diary file
    default_output = sys.stdout
    diary = open(diary_file,'w')
    sys.stdout = diary
    
    
    # running Ml2, could put in a try but doesn't matter
    MultiLevel2(resp_data,lvp_data, mat_file,outputspike_file,metadata1_file,\
              metdata2_file, gui_file,
                  spikefortemplate_file, uncurated_file, curated_file)
    
    
    # reset the sys.stdout to original settings
    sys.stdout = default_output
    
    # close the diary file
    diary.close()
    
    # if you want the diaryfile to print to screen
    # f = open(diary,'r')
    # print(f.read())
    # f.close()
    
    return
 
    


# Main Function

if __name__ == '__main__':
    
    
    # folder directory of matfiles
    # OK to have front slash here because added in dmf case
    # JOB: remove this feature!!!!
    dmf = ['../Animals/Animal_Data/Animal1/NeuralFiles/',
           '../Animals/Animal_Data/Animal2/NeuralFiles/'           
	   ]

    # in dir ML1
    # NO FRONT SLASH AT END: LOOK INSIDE: Programmatically done
    odm = ['../Animals/Animal_Data/Animal1/ML1_Output',
           '../Animals/Animal_Data/Animal2/ML1_Output'
	   ]

    # out dir ML2: NOTE if these exist it may be better to RENAME
    # at this stage the code will use a current Date Time for RENAME
    # NO FRONT SLASH AT END
    odm2 = ['../Animals/Animal_Data/Animal1/ML2_Output',
           '../Animals/Animal_Data/Animal2/ML2_Output'
	   ]

    # in file target  
    lvp = ['../Animals/Animal_Data/Animal1/TargetFiles/A1_LVP.mat',
           '../Animals/Animal_Data/Animal2/TargetFiles/A2_LVP.mat'
	    ]

    # in file target
    resp = ['../Animals/Animal_Data/Animal1/TargetFiles/A1_Resp.mat',
           '../Animals/Animal_Data/Animal2/TargetFiles/A2_Resp.mat'
	    ]       
    
    for dir_mat_files,out_dirML1,out_dirML2,lvp_file,resp_file in zip(dmf,odm,odm2,lvp,resp):
        try:
            os.mkdir(out_dirML2)
        except:
            #make name
            now = datetime.datetime.now()
            now = now.strftime('%Y-%m-%d_%H:%M:%S')
            out_dirML2 = out_dirML2 + now
            #make dir
            os.mkdir(out_dirML2)
        
        
        # Get the lvp/resp data, will be set to defaults if the files aren't there 
        lvp_data,resp_data = read_LvpResp_data(resp_file, lvp_file)
    
        # List of all processes running init
        process_list = []
            
        # for each channel Ml1 output and it's corresponding .mat file
        # sorted inde is ok since NeuralFiles names are equiv to ML1 out names
        if len(sorted(os.listdir(out_dirML1))) is not len(sorted(os.listdir(dir_mat_files))):
            print('fatal error: number NeuralFiles is not number of results in out_dirML1')
            continue

        for ch,neuralfile in zip(sorted(os.listdir(out_dirML1)),sorted(os.listdir(dir_mat_files))):
            
            # input files
            print('New iteration')
            # mat file is the directory + filename
            mat_file = dir_mat_files + neuralfile
            
            # from the channel folder, get the metadata/outputspike files
            # and add their directory infront of them such to give the filepath
            try:
                _ , metadataml1_file,outputspike_file = [out_dirML1 + '/' + ch + '/' + f \
                                                     for f in sorted(os.listdir(out_dirML1+ '/' + ch))]

                print(metadataml1_file)

            except Exception as E:
                #print E statement
                print(E)
                #label channel
                print('Error in ,',ch)
                #return to for and continue
                continue
    
            # Create the output folder for the specific Channel
            # Within the ML2 output folder
                
            ch_out_dirML2 = out_dirML2 + '/' + ch
            os.mkdir(ch_out_dirML2)
            
            # create the names of all the files with their path of ML2 output
            # format: Overall_Ml2_Output/Specific_Channel/file_ch.(extension)
            
            spikefortemplate_file = ch_out_dirML2 + '/' + 'SpikeForTemplate_' + ch + '.csv'
            gui_file = ch_out_dirML2 + '/' + 'GUI_' + ch + '.txt'
            metadataml2_file = ch_out_dirML2 + '/' + 'metadata_MultiLevel2_' + ch + '.txt'
            uncurated_file = ch_out_dirML2 + '/' + 'Uncurated_' + ch + '.csv' 
            curated_file = ch_out_dirML2 + '/' + 'Curated_' + ch + '.csv'
            diary_file = ch_out_dirML2 + '/' + 'diary_' + ch + '.txt'

            print(metadataml1_file)
            
            print('ML2 files created')
    
            #build process object
            p = Process(target = runChannelML2, \
                        args = (resp_data, lvp_data, metadataml1_file, mat_file, outputspike_file, diary_file, metadataml2_file, gui_file,\
                                spikefortemplate_file, uncurated_file, curated_file))
            
            # start process
            p.start()

            print('Process started')
            
            # place in started process list
            process_list.append(p)
            
            # used so the desktop doesn;t start force closin processes
            # no more than 5 processes run 
            while(len(multiprocessing.active_children()) == 5):
                  continue
        
        # ensures that the last processes get to finish before outputting
        # ML2 done in next line
        while(len(multiprocessing.active_children()) > 0):
                  continue
        
            
        print('MultiLevel2 Done')

