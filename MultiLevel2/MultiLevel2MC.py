#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d
from detect_peaks import detect_peaks

# In[ ]:


"""
process output from MultiLevel1: link spike/target uncurate/curate, prom/width bin, create templates
"""
__author__ = "Koustubh Sudarshan, Alex Karavos, Guy Kember"
__version__ = "1.0"
__license__ = "Apache v2.0"


# ## read/write helper functions for initialization

# ### read_Matlab_data

# In[ ]:


#load data
#neural
#U.keys() is top list: U['keyname'].keys: is member list
#['comment', 'interval', 'length', 'offset', 'scale', 'start', 'title', 'units', 'values']

#the resp/lvp is no longer used here can be taken out
def read_Matlab_data(mat_file,resp_file='',lvp_file = '',hasNEURAL=True, hasLVP=False, hasRESP=False) :
    
    try:
        if hasNEURAL :
            
            #load
            S = h5py.File(mat_file,'r')
    
            #abstract key list
            S_list = list([])
            for this_key in S :
                S_list.append(this_key)
                
            #channel number
            channel_number = S_list[0]
            
            #extract neural values: flatten for later cat
            xraw = np.array(S.get(S_list[0])['values']).flatten()
    
            #normalize xraw
            xraw = ( xraw - np.mean( xraw ) ) / np.std( xraw )
    
            #create times
            #sampling interval
            neural_interval = S.get(S_list[0])['interval'][0][0]
    
            #neural start time
            neural_start = S.get(S_list[0])['start'][0][0]
            
            #close input file
            S.close()
                
        else :
            print('Exception: hasNeural=',hasNEURAL)
            sys.exit()
    
        if hasLVP :
            
            #load
            T = h5py.File(lvp_file,'r')
            
            #abstract list
            T_list = list([])
            for this_key in T :
                T_list.append(this_key)
            
            #extract lvp
            #lvp values: : flatten for later cat
            lvpraw = np.array(T.get(T_list[0])['values']).flatten()
            
            #start time
            lvp_start = T.get(T_list[0])['start'][0][0]
            
            #lvp interval
            lvp_interval = T.get(T_list[0])['interval'][0][0]
            
            #close input file
            T.close()        
            
        else :
            print('Warning: read_Matlab_data: hasLVP=',hasLVP)
            #return empty
            lvpraw = np.array([])
            lvp_start = -9.0
            lvp_interval = -9.0
    
        #resp
        if hasRESP :
    
            #load
            U = h5py.File(resp_file,'r')
            
            #abstract list
            U_list = list([])
            for this_key in U :
                U_list.append(this_key)
    
            #extract respiratory
            #values: : flatten for later cat
            respraw = np.array(U.get(U_list[0])['values']).flatten()
    
            #start time
            resp_start = U.get(U_list[0])['start'][0][0]
            
            #resp interval
            resp_interval = U.get(U_list[0])['interval'][0][0]
            
            #close input file
            U.close()        
            
        else :
            print('Warning: read_Matlab_data: hasRESP=',hasRESP)
            #return empty
            respraw = np.array([])
            resp_start = -9.0
            resp_interval = -9.0
            
    
        return channel_number, xraw, neural_start, neural_interval, lvpraw, lvp_start, lvp_interval, respraw, resp_start, resp_interval

    except Exception as e:
        print(e)        
        print('Exception: read_Matlab_data')
        sys.exit()


# ### write MultiLevel2 metadata

# In[ ]:


def write_MultiLevel2_metadata(\
                               smoothing_width = 4,\
                               width_lower_bound=1.0,\
                               width_upper_bound=35.0,\
                               delta_prom=0.1,\
                               delta_width=1.0,\
                               before_max_ratio_bound=0.2,\
                               after_max_ratio_bound=0.3,\
                               min_spike_in_bin=10,\
                               hasLVP = False,\
                               hasRESP = False,\
                               lvpsmooth_local_max_win = 20,\
                               lvpsmooth_local_min_win = 10,\
                               neural_start = -9.0,\
                               neural_interval = -9.0,\
                               lvp_start = -9.0,\
                               lvp_interval = -9.0,\
                               resp_start = -9.0,\
                               resp_interval = -9.0,\
                               metadataml2_file = ''\
                               ) :

    f = open(metadataml2_file, 'w')

    #write parameters for MultiLevel2
    f.write('\nsmoothing_width\n')
    f.write(str(smoothing_width))
    
    f.write('width_lower_bound \n')
    f.write(str(width_lower_bound))

    f.write('\nwidth_upper_bound \n')
    f.write(str(width_upper_bound))

    f.write('\ndelta_prom (this is on the LOGARITHM) \n')
    f.write(str(delta_prom))

    f.write('\ndelta width \n')
    f.write(str(delta_width))

    f.write('\ndefault before_max_ratio_bound = 0.2 \n')
    f.write(str(before_max_ratio_bound))

    f.write('\ndefault after_max_ratio_bound = 0.3 \n')
    f.write(str(after_max_ratio_bound))

    f.write('\ndefault min_spike_in_bin = 10 \n')
    f.write(str(min_spike_in_bin))
    
    f.write('\nhasLVP \n')
    f.write(str(hasLVP))

    f.write('\nhasRESP \n')
    f.write(str(hasRESP))

    f.write('\nlvpsmooth_local_max_win \n')
    f.write(str(lvpsmooth_local_max_win))

    f.write('\nlvpsmooth_local_min_win \n')
    f.write(str(lvpsmooth_local_min_win))

    f.write('\n neural_start \n')
    f.write(str(neural_start))

    f.write('\n neural_interval \n')
    f.write(str(neural_interval))

    f.write('\n lvp_start \n')
    f.write(str(lvp_start))

    f.write('\n lvp_interval \n')
    f.write(str(lvp_interval))

    f.write('\n resp_start \n')
    f.write(str(resp_start))

    f.write('\n resp_interval \n')
    f.write(str(resp_interval))

    #close output file
    f.close()
    
    return


# ### read_MultiLevel1_spike

# In[ ]:


#read MultiLevel1 spike
#[channel_number, location_list_plus_total, location_list_minus_total]
def read_MultiLevel1_spike(outputspike_file) :

    # Create a dataframe from csv
    df = pd.read_csv(outputspike_file, delimiter=',')

    #sign index
    index = np.array(df['# index'])
    #get ' sign' ... do NOT leave out space before 'sign'!!!
    sign = np.array(df[' sign'])

    #get location_list_plus_total
    location_list_plus_total = index[np.nonzero(sign > 0)[0]]

    #get location_list_minus_total
    location_list_minus_total = index[np.nonzero(sign < 0)[0]]

    return location_list_plus_total, location_list_minus_total


# ### read_Multilevel1_metadata

# In[ ]:


# read MultiLevel_metadata
# [channel_number, location_list_plus_total, location_list_minus_total]
# channel_number, smoothing_width, pad, level_plus, level_minus, delta_level, min_level
# min_new_spike_plus,min_new_spike_minus, left, right, ring_threshold, ring_cutoff, ring_second,
# ring_num_period, min_peak_distance, mean_shift_n
    
def read_MultiLevel1_metadata(metadataml1_file,verbose=False) :

    #open read file
    print('Reading ML1 metadata')
    f = open(metadataml1_file, 'r')
    
    # extract lines with \n
    lines = [line for line in f]

    
    # close output
    f.close()
    
    # extract the variables we need
    for i in range(len(lines)):
        print(lines[i])
        if 'smoothing' in lines[i]: smoothing_width = int(lines[i+1])
        if 'left' in lines[i]: left = int(lines[i+1])
        if 'right' in lines[i]: right = int(lines[i+1])
        if 'mean_shift' in lines[i]: mean_shift_n = int(lines[i+1]) 


    #print result
    if verbose: print('smoothing width, left, right, mean_shift_n \n', smoothing_width, left, right, mean_shift_n)

    
    return int(smoothing_width), int(left), int(right), int(mean_shift_n)



# #### ourSmoothData - uses np.convolve - far faster than matlab translation of ourSmoothData
# ##### we need to use moving average equivalence to gaussian_1d for anything over window ~4 for speed

# In[ ]:


def ourSmoothData(values, halfwin) :
    window = 2 * halfwin + 1
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    sma = np.insert(sma,0,values[0:halfwin])
    sma = np.append(sma,values[-halfwin:])
    return sma


# ## helper functions for computation

# ### getAllPromWidth

# In[ ]:


def getAllPromWidth(\
                    x,\
                    location_list,\
                    mean_shift_n,\
                    width_lower_bound,\
                    left,\
                    right\
                    ) :

    ##############
    #init work var
    ##############
    
    #spike width
    spike_width = left + right
    
    #prominence
    prom = np.zeros(len(location_list))

    #half_width
    width = np.zeros(len(location_list))

    #spike locations in raw indexing
    spike_list = np.zeros(len(location_list)).astype(int)
    
    for spike_n in np.arange(len(location_list)) :

        #get spike index
        n = location_list[spike_n]
        
        #extract spike data
        xn = x[n - left : n + right]
        
        #shift spike mean
        xn = xn - np.mean(xn[0 : mean_shift_n])

        #set peak
        peak_max = xn[left]
        index_of_peak = left
        
        #get prom

        #min: flip x:, start=index_of_peak+2, mpd=width_lower_bound-1, default:first in flat spot
        [peak_min_array,_] = detect_peaks(-xn[index_of_peak + 2 : spike_width],mpd=width_lower_bound-1)           

        #if min
        if len(peak_min_array) > 0 :
            
            #reference to global index
            index_of_min = index_of_peak + 2 + peak_min_array[0]
            
            #store prom
            prom[spike_n] = peak_max - x[index_of_min]

            #store width
            width[spike_n] = index_of_min - index_of_peak
        
            #keep index
            spike_list[spike_n] = n
        #reject
        else :
            #reject index
            spike_list[spike_n] = -9
            
    #return
    if np.sum(spike_list > 0) > 0 :
        
        #have spike
        return prom[spike_list>0], width[spike_list>0], spike_list[spike_list>0]
    else :
        
        #have no spike
        print('WARNING: getAllPromWidth: no spike')
        return np.zeros(1), np.zeros(1), np.zeros(1).astype(int)
    


# ### getPromWidth

# In[ ]:


def getPromWidth(\
                 x,\
                 location_list,\
                 mean_shift_n,\
                 width_lower_bound,\
                 width_upper_bound,\
                 before_max_ratio_bound,\
                 after_max_ratio_bound,\
                 left,\
                 right\
                 ) :

    ##############
    #init work var
    ##############
    
    #spike width
    spike_width = left + right
    
    #prominence
    prom = np.zeros(len(location_list))

    #half_width
    width = np.zeros(len(location_list))

    #spike locations in raw indexing
    spike_list = np.zeros(len(location_list)).astype(int)

    for spike_n in np.arange(len(location_list)) :

        #get spike index
        n = location_list[spike_n]
        
        #extract spike data
        xn = x[n - left : n + right]
        
        #shift spike mean
        xn = xn - np.mean(xn[0 : mean_shift_n])

        #set peak
        peak_max = xn[left]
        index_of_peak = left

        #min: flip x:, start=index_of_peak+2, mpd=width_lower_bound-1, default:keep first in flat spot
        [peak_min_array,_] = detect_peaks(-xn[index_of_peak + 2 : spike_width], mpd=width_lower_bound-1)           

        #if min
        if len(peak_min_array) > 0 :
            
            #reference to global index
            index_of_min = index_of_peak + 2 + peak_min_array[0]
        
        #set to large width for rejection below
        else :
            index_of_min = spike_width - 5
        
        #prom
        prom_est = peak_max - xn[index_of_min]

        #width
        width_est = index_of_min - index_of_peak

        ###############
        #rejection test
        ###############

        #extract first half before peak
        halfway_to_peak = xn[0 : np.divmod(left,2)[0]]
        #print('halfway_to_peak ', halfway_to_peak)
        
        #test1: before zone: max abs value relative to peak       
        before_max_ratio = np.max( np.abs( halfway_to_peak ) ) / peak_max
        #print('before_max_ratio ', before_max_ratio)

        #test2: after zone: max abs value relative to peak
        after_max_ratio = np.abs(np.max( xn[index_of_peak + width_est : spike_width] ) / peak_max)
        #print('after_max_ratio', after_max_ratio)
        
        ##############################
        #decide if reject spike or not
        ##############################
        #print('before/after ratio, width, low/upp ',before_max_ratio, after_max_ratio, width_est, width_lower_bound, width_upper_bound)
        if before_max_ratio < before_max_ratio_bound and after_max_ratio < after_max_ratio_bound and\
           width_est > width_lower_bound and width_est < width_upper_bound :
            #print('***spike_n is ', spike_n)
            #keep prom
            prom[spike_n] = prom_est

            #keep half width
            width[spike_n] = width_est

            #keep index
            spike_list[spike_n] = n

    #return
    if np.sum(spike_list > 0) > 0 :
        
        #have spike
        return prom[spike_list>0], width[spike_list>0], spike_list[spike_list>0]
    
    else :
        
        #have no spike
        print('WARNING: getPromWidth: no spike')
        return np.zeros(1), np.zeros(1), np.zeros(1).astype(int)


# ### getPromWidthBin

# In[ ]:


def getPromWidthBin(\
                    prom,\
                    width,\
                    spike_list,\
                    delta_prom,\
                    delta_width\
                    ) :
    #set delta log prom and range
    log_prom = prom#np.log(prom)

    #set min
    min_log_prom = np.min(log_prom)
    min_width = np.min(width)

    #set max
    max_log_prom = np.max(log_prom)
    max_width = np.max(width)
    
    print('min_width ',min_width)

    print('max_width ',max_width)

    #set num center
    print(f'maxlogprom {max_log_prom}, minlogprom {min_log_prom}, deltaprom {delta_prom}')
    num_prom_bin = np.int((max_log_prom - min_log_prom)/delta_prom + 0.5) + 1
    num_width_bin = np.int((max_width - min_width)/delta_width + 0.5) + 1
            
    #set bin center
    prom_bin = min_log_prom + np.arange(num_prom_bin) * delta_prom
    width_bin = min_width + np.arange(num_width_bin) * delta_width
    
    #set num bin
    if num_prom_bin != len(prom_bin) or num_width_bin != len(width_bin) :
        print('WARNING: getPromWidthBin: unexpected num prom or width bin ')
    
    #construct 2d histogram: do ourselves for consistency with GUI apps and edge/bin control
    #2d histogram
    #prom_i, width_j vectors
    prom_width_count = np.zeros([num_prom_bin, num_width_bin]).astype(int)
    #zip: (prom_i, width_j, index): persistent iterator: list(zip())
    prom_width_data = list(zip(\
                               np.array((log_prom - min_log_prom)/delta_prom + 0.5).astype(int),\
                               np.array((width - min_width)/delta_width + 0.5).astype(int),\
                               np.arange(len(prom))\
                               )\
                            )

    #make 2dhistogram
    for (i, j, n) in prom_width_data :
        prom_width_count[i,j] = prom_width_count[i,j] + 1


    #3d array: list of spike in prom_width_count 2dhistogram
    #static array: oversize to rect: enable numpy ops: need max(prom_width_count) to build
    spike_in_bin = np.zeros([num_prom_bin, num_width_bin, np.int(np.max(prom_width_count) + 0.5)]).astype(int)

    #work var: clarity: track each bin
    spike_in_bin_count = np.zeros([num_prom_bin, num_width_bin]).astype(int)
    
    for (i, j, n) in prom_width_data :

        #work var: clarity
        count = spike_in_bin_count[i, j]

        #store index: prom_i(n), width_j(n), spike_list(n)
        spike_in_bin[i, j, count] = spike_list[n]
        
        #increment count in bin(i,j)
        spike_in_bin_count[i, j] = spike_in_bin_count[i, j] + 1        

    #IMPORTANT: want empty zeros AFTER nonzero raw spike indice
    #sort descending: flip(sort(ascending))
    spike_in_bin = np.flip(np.sort(spike_in_bin, axis=2), axis=2)

    #check for self consistency
    if np.max(spike_in_bin_count - prom_width_count ) != 0 :

        print('Warning: getPromWidthBin: error: spike in bin count ',\
                   'not equal to prom width count ')

    #collect output
    output = [min_log_prom, max_log_prom, min_width, max_width,\
              num_prom_bin, num_width_bin, prom_width_count, spike_in_bin]
    
    return  output


# ### getSpikeListCleanUp

# In[ ]:


def getSpikeListCleanUp(\
                        x,\
                        prom,\
                        width,\
                        num_prom_bin,\
                        num_width_bin,\
                        prom_width_count,\
                        spike_in_bin,\
                        mean_shift_n,\
                        spike_list,\
                        left,\
                        right\
                        ) :

    #work var
    spike_width = left + right
    
    #raw index to reject
    raw_index_to_reject = np.array([]).astype(int)

    #loop through prominence
    for i in np.arange(num_prom_bin) :

        #loop through width bin
        for j in np.arange(num_width_bin) :

            #note number of spike (prom i, width j) location
            num_of_indice = prom_width_count[i, j]
        
            #loop through spike at raw indice held in spike_in_bin
            #so long as there are at least 5 member
            if num_of_indice > 5 :
                #set_of_spike
                spike_in_this_bin = np.zeros([num_of_indice, spike_width])
 
                for k in np.arange(num_of_indice) :
        
                    #spike location in xraw
                    n = spike_in_bin[i, j, k]

                    #extract spike
                    xn = x[n - left : n + right]

                    #shift spike mean
                    xn = xn - np.mean(xn[0 : mean_shift_n])
                
                    #store xn
                    spike_in_this_bin[k,:] = xn
                
                #get median of spike_in_this_bin: iterate row axis: axis=0
                median_spike_in_this_bin = np.median(spike_in_this_bin, axis=0)
            
                #id spike indice where median_spike is high
                selected_index = np.nonzero(np.abs(median_spike_in_this_bin) > 1.0)[0]

                #sum: spike[selected index] > 3 * median_spike[selected_index]
                score_it = np.sum( np.abs(spike_in_this_bin[:, selected_index])
                                       > 3.0 * np.abs(median_spike_in_this_bin[selected_index]),\
                                      axis=1\
                                    )

                #reject spike_in_bin index if score exceeds threshold
                spike_in_bin_index_to_reject = np.nonzero(score_it > 10)[0]

                #set length of spike_in_bin_index_to_reject
                len_spike_in_bin_index_to_reject = len(spike_in_bin_index_to_reject)

                #rejecting spike and enough spike left over after reject removed
                if  len_spike_in_bin_index_to_reject > 0 and\
                    prom_width_count[i, j] - len_spike_in_bin_index_to_reject > 2 :
                    
                    #update prom_width_count
                    prom_width_count[i, j] = prom_width_count[i, j] - len_spike_in_bin_index_to_reject
                    
                    #find raw indice to reject
                    raw_index_to_reject = np.concatenate(\
                                                         [raw_index_to_reject,\
                                                          spike_in_bin[i, j, spike_in_bin_index_to_reject]]\
                                                          )
                    #0 reject spike: sort descending to place zero at end
                    spike_in_bin[i, j, spike_in_bin_index_to_reject] = 0
                    spike_in_bin[i, j, :] = np.flip(np.sort(spike_in_bin[i, j, :]))
                    
    #keep !reject spikes
    keep = np.nonzero(np.setdiff1d(spike_list, raw_index_to_reject))

    return  prom[keep], width[keep], spike_in_bin, spike_list[keep], prom_width_count


# ### getTemplate

# In[ ]:


def getTemplate(\
                x,\
                num_prom_bin,\
                num_width_bin,\
                prom_width_count,\
                spike_in_bin,\
                mean_shift_n,\
                min_spike_in_bin,\
                left,\
                right\
                ) :

    #work var
    spike_width = left + right
    
    #set_of_template(prom i, width j, raw data indice)
    set_of_template = np.zeros([num_prom_bin, num_width_bin, spike_width])
    
    #set of temporary spike in bin
    
    #loop through prominence
    for i in np.arange(num_prom_bin) :

        #loop through width bin
        for j in np.arange(num_width_bin) :

            #note number of spike (prom i, width j) location
            num_of_indice = prom_width_count[i, j]
        
            #determine if sufficient for significance
            if num_of_indice > min_spike_in_bin :
                
                #init template so ensure we have zeros
                template = np.zeros(spike_width)
                
                #loop through spike at raw indice
                #indice held in spike_in_bin
                for k in np.arange(num_of_indice) :

                    #spike location in xraw
                    n = spike_in_bin[i, j, k]

                    #extract spike
                    xn = x[n - left : n + right]

                    #shift spike mean
                    xn = xn - np.mean(xn[0 : mean_shift_n])
                    
                    #accumulate
                    template = template + xn 
                
                #set template as average
                template = template / num_of_indice
                
                #store template in set_of_template
                set_of_template[i, j, :] = template
                
    return set_of_template


# ### analysisLevel for Uncurated Spike

# In[ ]:


def analysisLevelUncurated(\
                           x,\
                           location_list,\
                           mean_shift_n,\
                           width_lower_bound,\
                           left,\
                           right\
                           ) :
        

   print('CALL: getAllPromWidth ')

   #use new variable spike_list since location_list is global to
   #calling routine: reject spike with WARNING: under NO WARNING 
   #condition: spike_list is same size as location_list
   [prom, width, spike_list] =   getAllPromWidth(\
       x,\
      location_list,\
       mean_shift_n,\
       width_lower_bound,\
       left,\
       right\
       )
                   
   #returns at least a single zero
   if len(spike_list) != 1 :
       print('\t have spike ',len(spike_list))
   
   else :
       print('\t have no spike ')
       #no spikes so send back nothing
       prom = np.zeros(1)
       width = np.zeros(1)
       spike_list = np.zeros(1).astype(int)
                      
   return prom, width, spike_list


# ### analysisLevel for Curated Spike

# In[ ]:


def analysisLevelCurated(\
     x,\
     location_list,\
     mean_shift_n,\
     width_lower_bound,\
     width_upper_bound,\
     before_max_ratio_bound,\
     after_max_ratio_bound,\
     left,\
     right\
     ) :
    
    
    print('CALL: getPromWidth ')

    #use new variable spike_list since location_list is global to
    #calling routine: reject spike with WARNING: under NO WARNING 
    #condition: spike_list is same size as location_list
    [prom, width, spike_list] =    getPromWidth(\
        x,\
        location_list,\
        mean_shift_n,\
        width_lower_bound,\
        width_upper_bound,\
        before_max_ratio_bound,\
        after_max_ratio_bound,\
        left,\
        right\
        )
                    

    if len(spike_list) != 1:
        print('\t have spike',len(spike_list))

    else :
        print('\t have no spike ')
        #no spikes so send back nothing
        prom = np.zeros(1)
        width = np.zeros(1)
        spike_list = np.zeros(1).astype(int)
 
    return prom, width, spike_list


# ### analysisLevel for Template

# In[ ]:


def analysisLevelTemplate(\
    x,\
    location_list,\
    mean_shift_n,\
    before_max_ratio_bound,\
    after_max_ratio_bound,\
    delta_prom,\
    delta_width,\
    width_lower_bound,\
    width_upper_bound,\
    min_spike_in_bin,\
    left,\
    right\
    ) :
    
    ######################################
    #start: set up spike list and template
    ######################################
    
    ############################
    #get level prominence/width
    #start here since removal of spike for clean
    #depends on knowledge of relative shape compared 
    #with spike in prom/width bin
    ############################
    print('CALL: getPromWidth ')

    [prom, width, spike_list] =\
        getPromWidth(\
            x,\
            location_list,\
            mean_shift_n,\
            width_lower_bound,\
            width_upper_bound,\
            before_max_ratio_bound,\
            after_max_ratio_bound,\
            left,\
            right\
            )
                    

    if len(spike_list) :
        print('\t spike for template has spike',len(spike_list))
    else :
        print('\t spike for template is empty ')
 
    #################################
    #get level prominence/width bin
    #################################
    if len(prom) > 10 :
        print('CALL: getPromWidthBin')
        [\
            min_log_prom,\
            max_log_prom,\
            min_width,\
            max_width,\
            num_prom_bin,\
            num_width_bin,\
            prom_width_count,\
            spike_in_bin\
            ] =\
            getPromWidthBin(\
                prom,\
                width,\
                spike_list,\
                delta_prom,\
                delta_width\
                )

        print('num prom/width ',num_prom_bin,num_width_bin)
 
        ########################
        #clean up spike in bin
        ########################


        print('CALL: getSpikeListCleanUp')

        [\
            prom,\
            width,\
            spike_in_bin,\
            spike_list,\
            prom_width_count\
            ] =\
            getSpikeListCleanUp(\
                x,\
                prom,\
                width,\
                num_prom_bin,\
                num_width_bin,\
                prom_width_count,\
                spike_in_bin,\
                mean_shift_n,\
                spike_list,\
                left,\
                right\
                )
        
        print('\t spikes for template: ',np.sum(prom_width_count))
        
        
        ##############
        #get template
        ##############
        print('CALL: getTemplate')

        set_of_template = getTemplate(\
            x,\
            num_prom_bin,\
            num_width_bin,\
            prom_width_count,\
            spike_in_bin,\
            mean_shift_n,\
            min_spike_in_bin,\
            left,\
            right\
            )

        #######################
        #end:   set up template
        #######################
    
    else :
        print('fewer than 10 spikes: no template')
        #no spikes so send back nothing
        min_log_prom = 0
        max_log_prom = 0
        min_width = 0
        max_width = 0
        num_prom_bin = 0
        num_width_bin = 0        
        prom = np.zeros(1)
        width = np.zeros(1)
        spike_list = np.zeros(1)
        prom_width_count = np.zeros(1)
        spike_in_bin = np.zeros([1,1,1])
        set_of_template = np.zeros([1,1,1])
    
    output = [\
        min_log_prom,\
        max_log_prom,\
        min_width,\
        max_width,\
        num_prom_bin,\
        num_width_bin,\
        prom,\
        width,\
        spike_list,\
        prom_width_count,\
        spike_in_bin,\
        set_of_template\
        ]
    return output


# ## helper functions for output

# ### get lvpphase

# In[ ]:
def getLvpPhase(lvp, diff_lvp) :

    ##############
    #set lvp phase
    ##############
    #normalize to zero mean and unit variance
    diff_lvp = (diff_lvp - np.mean( diff_lvp ) ) / np.std( diff_lvp )
    
    #init phase, systole (nearest int), diastole (nearest int)
    lvp_phase = np.zeros(len(diff_lvp)).astype(int)

    #set rising limb: 1.5 is generic
    index = np.nonzero(diff_lvp > 1.5)[0]
    lvp_phase[index] = 1
    
    #set falling limb: -1.5 is generic
    index = np.nonzero(diff_lvp < -1.5)[0]
    lvp_phase[index] = 3
    
    #correct form: 0000011110000033333300000111110000333000
    #trap errors: 010 101 313 131 030 303
    #[pattern, second difference, modify center value to]
    #010, -2, 0
    #101, 2, 1
    #313, 4, 3
    #131, -4, 1
    #030, -6, 0
    #303, 6, 3
    #make second diff list
    second_diff_list = np.array([-2,2,4,-4,-6,6])
    #make center_value_list
    center_value_list = np.array([0,1,3,1,0,3])
    #make fixed reference second_diff_lvp_phase
    second_diff_lvp_phase = np.diff(lvp_phase,2)
    
    for (this_diff, this_val) in zip(second_diff_list, center_value_list) :    
        index = np.nonzero(second_diff_lvp_phase==this_diff)[0]
        if len(index) > 0 :
            print('WARNING: getLvpPhase: increase lvpsmooth filter length:',this_diff, this_val)
            lvp_phase[index+1]=this_val

    #detect change array
    diff_lvp_phase = np.diff(lvp_phase)
    diff_lvp_phase = np.concatenate([diff_lvp_phase[0]*np.ones(1).astype(int),np.diff(lvp_phase)])

    #peak: 1->0 ... 0->3
    #index of diff pattern: 1->0
    d10=np.nonzero(diff_lvp_phase==-1)[0]
    #index of diff pattern: 0->3
    d03=np.nonzero(diff_lvp_phase[d10[0]+1:]==3)[0] + d10[0]+1
 
    #set peak phase: 2
    for (L,R) in zip(d10,d03) :
        #trap end where may have L < R
        if L < R :
            lvp_phase[L:R] = 2

    #min: 3->0 ... 0->1
    #index of diff pattern: 3->0
    
    
    d30 = np.nonzero(diff_lvp_phase==-3)[0]

    
    #index of diff pattern: 0->1: must follow 3->0: start beyond 0->1
    
    d01 = np.nonzero(diff_lvp_phase[d30[0]+1:]==1)[0]+d30[0]+1
    
    #set min phase: 4
    for (L,R) in zip(d30,d01) :
        #trap end where may have L < Rc
        if L < R :
            lvp_phase[L:R] = 4

    return lvp_phase

# ### get respphase

# In[ ]:


def getRespPhase(resp) :

    ##############
    #set resp phase
    ##############

    resp = ( resp - np.mean( resp ) ) / np.std( resp )
    
    #init phase to rest phase
    resp_phase = np.ones(len(resp)).astype(int)

    #rewrite: use zip in future
    for i in np.arange(201,len(resp)) :
        
        #subtraction mean pushes minimum down ~ -1.0 since the refractory
        #phase is 6/7 of the cycle - the laboratory equipment imposes
        #breathing over a 7 second cycle
        #a reduction by -0.25 is conservative
        if resp[i] > -0.25 :

            if resp[i-200] < resp[i] :
            
                resp_phase[i-100] = 2

            else :
                
                resp_phase[i-100] = 3
                
    

    return resp_phase


# ### write_MultiLevel2_spike

# In[ ]:


def write_MultiLevel2_spike(\
    allPrint,\
    channel_number,\
    x,\
    neural_start,\
    neural_interval,\
    lvpraw,\
    lvpsmooth,\
    diff_lvpsmooth,\
    respraw,\
    respsmooth,\
    mean_shift_n,\
    smoothing_width,\
    lvp_start,\
    lvp_interval,\
    resp_start,\
    resp_interval,\
    prom_plus,\
    width_plus,\
    spike_plus,\
    prom_minus,\
    width_minus,\
    spike_minus,\
    left,\
    right,\
    uncurated_file,\
    curated_file,\
    spikefortemplate_file\
    ) :
    
    
    #check valid allPrint
    if allPrint != -1 and allPrint != 0 and allPrint != 1 :
        
        print('Exception: bad allPrint: do not know where to print', allPrint)
        
        return


    if spike_plus[0] > 0 and spike_minus[0] > 0 :

        #set spike_for_template
        spike = np.concatenate([spike_plus, spike_minus])

        #set prom
        prom = np.concatenate([prom_plus, prom_minus])

        #set width
        width = np.concatenate([width_plus, width_minus])
                            
        #set plus and minus identification
        len_spike_plus = len(spike_plus)
        len_spike_minus = len(spike_minus)
        plus_id = np.ones(len_spike_plus).astype(int)
        minus_id = -1 * np.ones(len_spike_minus).astype(int)
        plus_minus_id = np.concatenate([plus_id, minus_id])

    elif spike_plus[0] > 0 :
        
        #set spike_list
        spike = spike_plus

        #set prom
        prom = prom_plus

        #set width
        width = width_plus

        #set plus_minus identification
        len_spike_plus = len(spike_plus)
        plus_id = np.ones(len_spike_plus).astype(int)
        plus_minus_id = plus_id       
        
    elif spike_minus[0] > 0 :

        #set spike_list
        spike = spike_minus
        
        #set prom
        prom = prom_minus

        #set width
        width = width_minus

        #set plus_minus identification
        len_spike_minus = len(spike_minus)
        minus_id = -1 * np.ones(len_spike_minus).astype(int)
        plus_minus_id = minus_id       

    else :
        
        print('\n Exception: getOutputSpike: no spikes ')

        return
    
    #get index for time_stamp increasing
    #sort spike (which is raw index) into ascending order
    #and retain sort index
    index = np.argsort(spike)
    
    #sort spike (can do in previous line but this is clearer)
    spike = spike[index]

    #set spike time at spike index
    spike_time = neural_start + neural_interval * spike.astype(float)
    
    #spike raw x level
    spike_x_level = x[spike]

    #sort plus minus id
    plus_minus_id = plus_minus_id[index]

    #sort prom
    prom = prom[index]

    #sort width
    width = width[index]

    #overlap: neural, lvp, resp
    #start time
    print('neural/lvp/resp start ',neural_start, lvp_start, resp_start)
    
    start_time = np.max([neural_start, lvp_start, resp_start])
    
    print('start time ',start_time)
    
    #end time
    neural_end = neural_start + np.max(spike) * neural_interval
    lvp_end = lvp_start + np.max(len(lvpraw)) * lvp_interval
    resp_end = resp_start + np.max(len(respraw)) * resp_interval
    
    print('neural/lvp/resp end ',neural_end, lvp_end, resp_end)
    
    end_time = np.min([end for end in [neural_end, lvp_end, resp_end] if end > 0.0])

    print('end time ',end_time)
    
    #impose bounds: avoid edges
    index = np.nonzero(np.logical_and(spike_time > start_time, spike_time < end_time))[0]
    #filter: spike, spike_time, spike_x_level, prom, width
    spike = spike[index]
    spike_time = spike_time[index]
    spike_x_level = spike_x_level[index]
    plus_minus_id = plus_minus_id[index]
    prom = prom[index]
    width = width[index]
    
    ###########
    #set up lvp
    ###########

    #have lvp
    if len(lvpraw) > 0 :
        #determine associated lvp sample index: len(lvp) = len(spike_time)
        lvp_index = np.array((spike_time - lvp_start) / lvp_interval + 0.5).astype(int)

        #check bounds
        if np.sum(lvp_index < 0) > 0 :
            print('list ',np.nonzero(lvp_index<0)[0])
            print('WARNING: write_MultiLevel2_spike: LVP: bad time overlap')

        #get lvpraw when neuron sampled        
        lvp = lvpraw[lvp_index]

        #################
        #set up lvp phase
        #################
        #get phase at neural sample time

        lvp_phase = getLvpPhase(lvpsmooth, diff_lvpsmooth)
        lvp_phase = lvp_phase[lvp_index] 

    #do not have lvp
    else :
        lvp = np.ones(len(index)) * -9.0
        lvp_phase = np.ones(len(index)).astype(int) * -9

    ############
    #set up resp
    ############

    #have resp
    if len(respraw) > 0 :
        #determine associated resp sample index
        resp_index = np.array((spike_time - resp_start) / resp_interval + 0.5).astype(int)

        #check bounds
        if np.sum(resp_index < 0) > 0 :
            print('WARNING: write_MultiLevel2_spike: RESP: bad time overlap')
 
        #get respraw when neuron sampled
        resp = respraw[resp_index]
        
        ##################
        #set up resp phase
        ##################

        #get phase at neural sample time
        resp_phase = getRespPhase(respsmooth)
        resp_phase = resp_phase[resp_index]        

    #do not have resp
    else :
        resp = np.ones(len(index)) * -9.0
        resp_phase = np.ones(len(index)).astype(int) * -9


    #output uncurated spike
    if allPrint == -1 :

        outputFilename = uncurated_file

    #output curated spike
    elif allPrint == 0 :

        outputFilename = curated_file

    #output spike for template
    elif allPrint == 1 :

        outputFilename = spikefortemplate_file


    df = pd.DataFrame(\
        {'spike_time':spike_time,\
         'spike':spike,\
         'lvp':lvp,\
         'lvp_phase':lvp_phase,\
         'resp':resp,\
         'resp_phase':resp_phase,\
         'plus_minus_id':plus_minus_id,\
         'prom':prom,\
         'width':width,\
         'spike_x_level':spike_x_level,\
         }\
         )
    df.to_csv(outputFilename, index=False, mode='a')

    return


# In[ ]:


def write_MultiLevel2_GUI(\
    channel_number,\
    mean_shift_n,\
    smoothing_width,\
    min_log_prom_plus,\
    max_log_prom_plus,\
    min_width_plus,\
    max_width_plus,\
    num_prom_bin_plus,\
    num_width_bin_plus,\
    prom_width_count_plus,\
    spike_in_bin_plus,\
    set_of_template_plus,\
    min_log_prom_minus,\
    max_log_prom_minus,\
    min_width_minus,\
    max_width_minus,\
    num_prom_bin_minus,\
    num_width_bin_minus,\
    prom_width_count_minus,\
    spike_in_bin_minus,\
    set_of_template_minus,\
    left,\
    right,gui_file) :
   
    #work var
    spike_width = left + right
                    
    #init
    have_plus_template = False
    have_minus_template = False
    
    [plus_x, plus_y, _] = set_of_template_plus.shape
    [minus_x, minus_y, _] = set_of_template_minus.shape
    
    if plus_x > 1 and plus_y > 1 :

        #have plus template
        have_plus_template = True

    if minus_x > 1 and minus_y > 1 :

        #have minus template
        have_minus_template = True
        
    if not have_plus_template and not have_minus_template :

        #send warning
        print('Exception: write_MultiLevel2_GUI: no template')

        return

    #open output file
    fId = open(gui_file,'w')

    #output channel number and mean_shift
    fId.write('channel_number \n')
    fId.write(str(channel_number))
    fId.write('\nsmoothing_width \n')
    fId.write(str(smoothing_width))
    fId.write('\nmean_shift_n\n')
    fId.write(str(mean_shift_n))

    #output number of template plus
    fId.write('\nnumber of template plus: num_prom_bin_plus * num_width_bin_plus\n')
    fId.write(str(num_prom_bin_plus * num_width_bin_plus))

    #output plus bin information
    fId.write('\nmin_log_prom_plus \n')
    fId.write(str(min_log_prom_plus))
    fId.write('\nmax_log_prom_plus \n')
    fId.write(str(max_log_prom_plus))
    fId.write('\nmin_width_plus \n')
    fId.write(str(min_width_plus))
    fId.write('\nmax_width_plus \n')
    fId.write(str(max_width_plus))
    fId.write('\nnum_prom_bin_plus \n')
    fId.write(str(num_prom_bin_plus))
    fId.write('\nnum_width_bin_plus \n')
    fId.write(str(num_width_bin_plus))

    #output number of template minus
    fId.write('\nnumber of template minus: num_prom_bin_minus * num_width_bin_minus\n')
    fId.write(str(num_prom_bin_minus * num_width_bin_minus))

    #output minus bin information
    fId.write('\nmin_log_prom_minus\n')
    fId.write(str(min_log_prom_minus))
    fId.write('\nmax_log_prom_minus\n')
    fId.write(str(max_log_prom_minus))
    fId.write('\nmin_width_minus\n')
    fId.write(str(min_width_minus))
    fId.write('\nmax_width_minus\n')
    fId.write(str(max_width_minus))
    fId.write('\nnum_prom_bin_minus\n')
    fId.write(str(num_prom_bin_minus))
    fId.write('\nnum_width_bin_minus\n')
    fId.write(str(num_width_bin_minus))

    #################
    #do plus template
    #################

    #loop through prominence bin: use write(.join(map(str,dataset))) in future
    for i in np.arange(num_prom_bin_plus) :

        #loop through width bin
        for j in np.arange(num_width_bin_plus) :

            #output number of bin member
            fId.write(str(i) + ' ' + str(j) + ' ' + str(prom_width_count_plus[i,j]) + ' ')

            #output bin member global index - this connect back
            #to spike output file
            for k in np.arange(prom_width_count_plus[i,j]) :

                fId.write(str(spike_in_bin_plus[i,j,k]) + ' ')

            #output bin template
            for k in np.arange(spike_width) :

                fId.write(str(set_of_template_plus[i, j, k]) + ' ')

            fId.write('\n')


    ##################
    #do minus template
    ##################

    #loop through prominence bin
    for i in np.arange(num_prom_bin_minus) :

        #loop through width bin
        for j in np.arange(num_width_bin_minus) :

            #output number of bin member
            fId.write(str(i) + ' ' + str(j) + ' ' + str(prom_width_count_minus[i,j]) + ' ')

            #output bin member global index - this connect back
            #to spike output file
            for k in np.arange(prom_width_count_minus[i,j]) :

                fId.write(str(spike_in_bin_minus[i,j,k]) + ' ')

            #output bin template
            for k in np.arange(spike_width) :

                fId.write(str(set_of_template_minus[i, j, k]) + ' ')

            fId.write('\n')



    return


# In[ ]:
# ## MultiLevel2 Main

def MultiLevel2(resp_data,\
                lvp_data,\
                mat_file,\
                outputspike_file,\
                metadataml1_file,\
                metadataml2_file,\
                gui_file,\
                spikefortemplate_file,\
                uncurated_file,\
                curated_file,\
                ch = 'ch#Unknown',\
                smoothing_width = 4,\
                width_lower_bound = 1.0,\
                width_upper_bound = 35.0,\
                delta_prom = 0.1,\
                delta_width = 1.0,\
                before_max_ratio_bound = 0.2,\
                after_max_ratio_bound = 0.3,\
                min_spike_in_bin = 10,\
                hasLVP = False,\
                hasRESP = False,\
                lvpsmooth_local_max_win = 20,
                lvpsmooth_local_min_win = 10):

    #load data
    [channel_number,\
     xraw,\
     neural_start,\
     neural_interval,\
     _,_,_,_,_,_] = read_Matlab_data(mat_file = mat_file)

    lvpraw,lvpsmooth,diff_lvpsmooth,lvp_start,lvp_interval = lvp_data
    respraw,respsmooth,resp_start,resp_interval = resp_data

    ch = mat_file.split('_')[-1][:-4]
    channel_number = ch
    
    print('*****neural_start******* ',neural_start)

    #read MultiLevel1 metadata
    [smoothing_width, left, right, mean_shift_n] = read_MultiLevel1_metadata(metadataml1_file)
    print('smooth left right mean shift ', smoothing_width, left, right, mean_shift_n)

    #read MultiLevel1 spike
    location_list_plus_total, location_list_minus_total = read_MultiLevel1_spike(outputspike_file)
    print('list plus minus ',len(location_list_plus_total), len(location_list_minus_total))

    #up to 4 is okay for speed
    xsmooth = gaussian_filter1d(xraw, smoothing_width)

    #write MultiLevel2 metadata

    write_MultiLevel2_metadata(\
       smoothing_width = smoothing_width,\
       width_lower_bound = width_lower_bound,\
       width_upper_bound= width_upper_bound,\
       delta_prom=delta_prom,\
       delta_width=delta_width,\
       before_max_ratio_bound=before_max_ratio_bound,\
       after_max_ratio_bound=after_max_ratio_bound,\
       min_spike_in_bin=min_spike_in_bin,\
       hasLVP = hasLVP,\
       hasRESP = hasRESP,\
       lvpsmooth_local_max_win = lvpsmooth_local_max_win,\
       lvpsmooth_local_min_win = lvpsmooth_local_min_win,\
       neural_start = neural_start,\
       neural_interval = neural_interval,\
       lvp_start = lvp_start,\
       lvp_interval = lvp_interval,\
       resp_start = resp_start,\
       resp_interval = resp_interval,\
        metadataml2_file=metadataml2_file\
               )

    # Smoothing Was here but was removed

    print(' ***************************************************************')
    print(' *********START UNCURATED SPIKE LIST ***************************')
    print(' ***************************************************************')

    #plus: get spike prom width
    [uncuratedProm_plus,uncuratedWidth_plus,uncuratedSpike_plus] =\
        analysisLevelUncurated(xsmooth,location_list_plus_total,mean_shift_n,width_lower_bound,left,right)

    #minus: get spike prom width
    [uncuratedProm_minus,uncuratedWidth_minus,uncuratedSpike_minus] =\
        analysisLevelUncurated(-xsmooth,location_list_minus_total,mean_shift_n,width_lower_bound,left,right)

    print(' **************************************************')
    print(' CALL: getOutput Uncurated Spike                   ')
    print(' **************************************************')

    #uncurated allPrint=-1
    allPrint = -1
    write_MultiLevel2_spike(allPrint,\
        channel_number,\
        xsmooth,\
        neural_start,\
        neural_interval,\
        lvpraw,\
        lvpsmooth,\
        diff_lvpsmooth,\
        respraw,\
        respsmooth,\
        mean_shift_n,\
        smoothing_width,\
        lvp_start,\
        lvp_interval,\
        resp_start,\
        resp_interval,\
        uncuratedProm_plus,\
        uncuratedWidth_plus,\
        uncuratedSpike_plus,\
        uncuratedProm_minus,\
        uncuratedWidth_minus,\
        uncuratedSpike_minus,\
        left,\
        right,\
        uncurated_file,\
        curated_file,\
        spikefortemplate_file\
        )

    print(' *************************************************************')
    print(' *********START CURATED SPIKE LIST AND PROM WIDTH ************')
    print(' *************************************************************')

    #plus: get spike prom width
    [curatedProm_plus,curatedWidth_plus,curatedSpike_plus] =\
        analysisLevelCurated(\
            xsmooth,location_list_plus_total,\
            mean_shift_n,\
            width_lower_bound,\
            width_upper_bound,\
            before_max_ratio_bound,\
            after_max_ratio_bound,\
            left,\
            right\
            )

    #minus: get spike prom width
    [curatedProm_minus,curatedWidth_minus,curatedSpike_minus] =\
        analysisLevelCurated(\
            -xsmooth,\
            location_list_minus_total,\
            mean_shift_n,\
            width_lower_bound,\
            width_upper_bound,\
            before_max_ratio_bound,\
            after_max_ratio_bound,\
            left,\
            right\
            )

    print(' ******************************************************')
    print(' CALL: getOutput Curated Spike and Target                  ')
    print(' ******************************************************')

    #curated allPrint=0
    allPrint = 0
    write_MultiLevel2_spike(allPrint,\
        channel_number,\
        xsmooth,\
        neural_start,\
        neural_interval,\
        lvpraw,\
        lvpsmooth,\
        diff_lvpsmooth,\
        respraw,\
        respsmooth,\
        mean_shift_n,\
        smoothing_width,\
        lvp_start,\
        lvp_interval,\
        resp_start,\
        resp_interval,\
        curatedProm_plus,\
        curatedWidth_plus,\
        curatedSpike_plus,\
        curatedProm_minus,\
        curatedWidth_minus,\
        curatedSpike_minus,\
        left,\
        right,\
        uncurated_file,\
        curated_file,\
        spikefortemplate_file\
        )


    print(' ***************************************************')
    print(' *********START SPIKE LIST AND TEMPLATE ************')
    print(' ***************************************************')
    print('')

    #plus: get spike and template
    [min_log_prom_plus,\
    max_log_prom_plus,\
    min_width_plus,\
    max_width_plus,\
    num_prom_bin_plus,\
    num_width_bin_plus,\
    prom_plus,\
    width_plus,\
    spike_plus,\
    prom_width_count_plus,\
    spike_in_bin_plus,\
    set_of_template_plus\
    ] =\
        analysisLevelTemplate(\
            xsmooth,\
            location_list_plus_total,\
            mean_shift_n,\
            before_max_ratio_bound,\
            after_max_ratio_bound,\
            delta_prom,delta_width,\
            width_lower_bound,\
            width_upper_bound,\
            min_spike_in_bin,\
            left,\
            right\
            )


    #minus: get spike and template
    [min_log_prom_minus,\
    max_log_prom_minus,\
    min_width_minus,\
    max_width_minus,\
    num_prom_bin_minus,\
    num_width_bin_minus,\
    prom_minus,\
    width_minus,\
    spike_minus,\
    prom_width_count_minus,\
    spike_in_bin_minus,\
    set_of_template_minus\
    ] =\
        analysisLevelTemplate(\
            -xsmooth,\
            location_list_minus_total,\
            mean_shift_n,\
            before_max_ratio_bound,\
            after_max_ratio_bound,\
            delta_prom,\
            delta_width,\
            width_lower_bound,\
            width_upper_bound,\
            min_spike_in_bin,\
            left,\
            right\
            )

    print(' ******************************************************')
    print(' CALL: getOutput SpikeForTemplate and Target           ')
    print(' ******************************************************')

    #spikeForTemplate allPrint=1
    allPrint = 1
    write_MultiLevel2_spike(\
        allPrint,\
        channel_number,\
        xsmooth,\
        neural_start,\
        neural_interval,\
        lvpraw,\
        lvpsmooth,\
        diff_lvpsmooth,\
        respraw,\
        respsmooth,\
        mean_shift_n,\
        smoothing_width,\
        lvp_start,\
        lvp_interval,\
        resp_start,\
        resp_interval,\
        prom_plus,\
        width_plus,\
        spike_plus,\
        prom_minus,\
        width_minus,\
        spike_minus,\
        left,\
        right,\
        uncurated_file,\
        curated_file,\
        spikefortemplate_file\
        )


    print(' **********************************************')
    print(' CALL: getOutput Bin Template Spike for GUI    ')
    print(' **********************************************')

    #output for GUI
    write_MultiLevel2_GUI(\
    channel_number,\
        mean_shift_n,\
        smoothing_width,\
        min_log_prom_plus,\
        max_log_prom_plus,\
        min_width_plus,\
        max_width_plus,\
        num_prom_bin_plus,\
        num_width_bin_plus,\
        prom_width_count_plus,\
        spike_in_bin_plus,\
        set_of_template_plus,\
        min_log_prom_minus,\
        max_log_prom_minus,\
        min_width_minus,\
        max_width_minus,\
        num_prom_bin_minus,\
        num_width_bin_minus,\
        prom_width_count_minus,\
        spike_in_bin_minus,\
        set_of_template_minus,\
        left,\
        right,\
        gui_file\
        )
    print(' ***************************************************')
    print(' *********SPIKE LIST AND TEMPLATE COMPLETE**********')
    print(' ***************************************************')

