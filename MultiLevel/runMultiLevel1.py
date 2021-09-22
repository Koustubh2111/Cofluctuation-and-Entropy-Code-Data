#!/usr/bin/env python
# coding: utf-8

# In[2]:
import os
from MultiLevel1MC import MultiLevel1
import pandas as pd
import sys
from multiprocessing import Process
import time
import multiprocessing



# function made to run processes of ML1
# takes in the names of the input matfile, and the output files
            
def runChannelML1(matfile = '',outputspikefile='',metadatafile = '',diaryfile = ''):
    
    os.nice(10)
    #print('Entering runChannel1ML1 with ch = ',ch)
   
    # get all ML1 parameters from inputfile.csv
    # This may be generating an error, going to remove for now
    #smoothing_width,level_plus_factor,level_minus_factor,delta_level,min_level,min_new_spike_plus,min_new_spike_minus, \
    #left,right,ring_threshold,ring_cutoff,ring_second,ring_num_period,\
    #mpd,mean_shift_n = inputParams(i)
    

    #Create diary file and make all printing go there
    #save the default printing settings
    defaultout = sys.stdout
    #Make a DiaryFile where all the printing will go
    diary = open(diaryfile,'w')
    sys.stdout = diary
    
    

    try:
        
        #print('entering multilevel')
        MultiLevel1(matfile,outputspikefile,metadatafile)   
        pass
    
    except Exception as e:
        print(matfile,outputspikefile,metadatafile)
        print('Exception:\n\n\n',e)
        sys.exit()
    finally:
        pass
    
    sys.stdout =  defaultout
    diary.close()
    
    #Print out the diary file to screen
    #f =  open(diary,'r')
    #print(f.read())
    #f.close()      

    return



# In[27]:
#Required to run from cmd to us Process
if __name__ == '__main__':
    
    os.nice(10)
    #List of directories containing the spike data for all channels for TWO animals as an example 
    dmf = ['../HeartFailureAnimals/pig1666/NeuralFiles/',\
           '../HeartFailureAnimals/pig1767/NeuralFiles/']
            
    #List of output directories for the two animals to store the results    
    odm = ['../HeartFailureAnimals/pig1666/ML1_Output',\
           '../HeartFailureAnimals/pig1767/ML1_Output']    
            

    timep = time.time()
    
    n = 0
    #Going through each animal
    for dir_mat_files,out_dir in zip(dmf,odm):
        print(dir_mat_files,'\n',out_dir,'\n\n\n')

        t = time.time()
        # Try making the output Results directory
        try:
            os.mkdir(out_dir)
            pass
        except:
            out_dir = out_dir+ str(time.time()) 
            os.mkdir(out_dir)
            #print('Result Directory already existed, renamed as', out_dir)
        finally:
            pass       
    

        # List of neural files
        #Contains the spike data in .mat for all channels. Two channels for each animal is shown here
        neural_folder = sorted(os.listdir(dir_mat_files))
        # print(f'There are {len(neural_file_list)} listed')
         
        # time counter started
    
        # List of all processes
        
        for neuralfile in neural_folder:
            
            n = n+1
            # Check to make sure directory has .mat files
            # Skip the file if it isn't a .mat file
            
            if neuralfile[-4:] != '.mat':
                print(f"There's a file named {neuralfile} that isn't a .mat file")
                continue
            
            # !!!!!!!!!!!!!!!!! #
            # !!!!!!!!!!!!!!!!! #
    
            # This line needs to be set depending on the input file format 
            # example file name 'neural_icn17.mat'
            
            ch = neuralfile[:-4] # would return pig1740_icn17
     
            # !!!!!!!!!!!!!!!!! #
            # !!!!!!!!!!!!!!!!! #
            
            # Create the folder in the output directory
            # note, this shouldn't ever draw an error because the 'out_dir' made 
            # was unique, or unique with a timestamp
            
            out_dir_ch = out_dir+ '/' + ch + '/'
            os.mkdir(out_dir_ch)
            
            # the matfile is the neuralfile with the directory attached
            matfile = dir_mat_files + neuralfile
            
            #print('matfile =',matfile,' \n\n\n\n\n')
            
            # create what the outputfiles will look like:
            # for example:
            # .../research/multichannelresults/icn17/outputSpike_icn17.csv   
            
            outputspikefile = out_dir_ch + 'outputSpike_' + ch + '.csv'
            metadatafile = out_dir_ch + 'metadataMultilevel1_' + ch + '.txt'
            diaryfile = out_dir_ch + 'diaryML1_' + ch + '.txt'
            
            
            # create process
            # name is what the process is refered to if printed
            # target is the target function that the process will execute
            
            p = Process(name = neuralfile, target = runChannelML1,args = (matfile,outputspikefile,metadatafile,diaryfile))
            
            # start the process
            p.start( )
            
            #Only keep adding processes if there is less than five running
            while(len(multiprocessing.active_children()) == 4):
                  continue
              
            # END for loop         
        
        process_list = multiprocessing.active_children()

        for p in process_list:
        # this causes this function to wait until all the processes are done
            p.join()
        
        print(f'{out_dir} Done: Time taken:')
            
        print(round(time.time()-t,2)) 
        
        # used for measuring the speed in seconds
        # print(f'time parallel was {round(time_par)}')
    print(f'{round(time.time()-timep,1)} for {n} channels to be run')
    
    
         
