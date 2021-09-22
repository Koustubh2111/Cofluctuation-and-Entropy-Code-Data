#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter1d
from detect_peaks import detect_peaks

# # ported from MexSoftware_2 normalityMultiLevel1


# In[ ]:


"""
process output from MultiLevel1: link spike/target uncurate/curate, prom/width bin, create templates
"""
__author__ = "Koustubh Sudarshan, Alex Karavos, Guy Kember"
__version__ = "1.0"
__license__ = "Apache v2.0"

# ## helper functions

# ### read_Matlab_data

# In[ ]:


#load data
#neural
#['comment', 'interval', 'length', 'offset', 'scale', 'start', 'title', 'units', 'values']
def read_Matlab_data(matfile = 'neural.mat',hasNEURAL=True, hasLVP=False, hasRESP=False) :
    
    if hasNEURAL :
        try:        
            #load
            S = h5py.File(matfile,'r')
    
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
    
            #sampling interval
            neural_interval = S.get(S_list[0])['interval'][0][0]
    
            #neural start time
            neural_start = S.get(S_list[0])['start'][0][0]
            
            #close input file
            S.close()
        except Exception as e:
            print(e)
            sys.exit()
            
    else :
        print('Exception: hasNeural=',hasNEURAL)
        sys.exit()

    if hasLVP :
        
        #load
        T = h5py.File('lvp.mat','r')
        
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
        U = h5py.File('resp.mat','r')
        
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
        resp_interval = U.get(U_list[0])['interval']

        #close input file
        U.close()        
        
    else :
        print('Warning: read_Matlab_data: hasRESP=',hasRESP)
        #return empty
        respraw = np.array([])
        resp_start = -9.0
        resp_interval = -9.0

        return channel_number, xraw, neural_start, neural_interval, lvpraw, lvp_start, lvp_interval, respraw, resp_start, resp_interval
    


# ### write_MultiLevel1_data

# In[ ]:


#numpy.savetxt: write data to txt file
def write_MultiLevel1_spike(
                location_list_plus_total,\
                location_list_minus_total,\
                outputSpike
                ) :
    #create sign    
    #plus: [k,]
    plus = np.ones(len(location_list_plus_total)).astype(int)    
    #minus: [l,]
    minus = np.ones(len(location_list_minus_total)).astype(int) * -1
    
    #cat sign
    #reshape to [1,k+l] array from vector
    sign = np.concatenate([plus, minus]).reshape(1, len(plus)+len(minus))    
    #reshape to [1,k+l] array from vector
    index = np.concatenate([location_list_plus_total, location_list_minus_total]).reshape(1, len(plus)+len(minus))

    #cat set
    #cat to [2, k+l].transpose = [k+l,2]
    index_and_sign = np.concatenate([index, sign]).transpose()
    
    #save numpy array to csv
    np.savetxt(outputSpike, index_and_sign, delimiter=',', header='index, sign', fmt='%d')

    return


# ### write MultiLevel metadata

# In[ ]:


#numpy.savetxt: write data to txt file
def write_MultiLevel1_metadata(metaData = 'metadata_MultiLevel1.txt',\
                               channel_number='ICNunknown',\
                              smoothing_width=4,\
                              pad=500,\
                              level_plus_factor=0.9,\
                              level_minus_factor=0.8,\
                              delta_level=0.1,\
                              min_level=0.8,\
                              min_new_spike_plus=1000,\
                              min_new_spike_minus=500,\
                              left=20,\
                              right=100,\
                              ring_threshold=0.5,\
                              ring_cutoff=0.5,\
                              ring_second=0.06,\
                              ring_num_period=5,\
                              mpd=60,\
                              mean_shift_n=5,
                              neural_start = -9.0,\
                              neural_interval = -9.0,\
                              lvp_start = -9.0,\
                              lvp_interval = -9.0,\
                              resp_start = -9.0,\
                              resp_interval = -9.0\
                              ) :

    f = open(metaData, 'w')

    f.write('channel number \n')
    f.write(channel_number)

    f.write('\n smoothing width  \n')
    f.write(str(smoothing_width))

    f.write(' \n pad \n')
    f.write(str(pad))

    f.write(' \n level_plus_factor \n')
    f.write(str(level_plus_factor))

    f.write(' \n level_minus_factor \n')
    f.write(str(level_minus_factor))

    f.write(' \n delta_level \n')
    f.write(str(delta_level))

    f.write(' \n min_level \n')
    f.write(str(min_level))

    f.write(' \n min_new_spike_plus \n')
    f.write(str(min_new_spike_plus))

    f.write(' \n min_new_spike_minus \n')
    f.write(str(min_new_spike_minus))

    f.write(' \n left \n')
    f.write(str(left))

    f.write(' \n right \n')
    f.write(str(right))

    f.write(' \n ring_threshold \n')
    f.write(str(ring_threshold))

    f.write(' \n ring_cutoff \n')
    f.write(str(ring_cutoff))

    f.write(' \n ring_second \n')
    f.write(str(ring_second))

    f.write(' \n ring_num_period \n')
    f.write(str(ring_num_period))

    f.write(' \n min peak distance: mpd \n')
    f.write(str(mpd))

    f.write(' \n mean_shift_n \n')
    f.write(str(mean_shift_n))

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
    
    f.close()
    return


# ### read_MultiLevel1_data

# In[ ]:


#read MultiLevel1 data
#[channel_number, location_list_plus_total, location_list_minus_total]
def read_MultiLevel1_data(outputSpike = 'outputSpike.csv') :

    # Create a dataframe from csv
    df = pd.read_csv(outputSpike, delimiter=',')

    #sign index
    index = np.array(df['0'])
    #get sign
    sign = np.array(df['1'])
    
    #get location_list_plus_total
    location_list_plus_total = index[np.nonzero(sign > 0)[0]]

    #get location_list_minus_total
    location_list_minus_total = index[np.nonzero(sign < 0)[0]]

    return location_list_plus_total, location_list_minus_total


# ### ourSmoothData 
# #### use np.convolve
# ##### need to use moving average equivalence to gaussian_1d for  window >4 for speed

# In[ ]:


def ourSmoothData(values, halfwin) :
    window = 2 * halfwin + 1
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    sma = np.insert(sma,0,values[0:halfwin])
    sma = np.append(sma,values[-halfwin:])
    return sma


# In[ ]:
#get new level that satisfies number of spike constraint
#vectorized
def getNewLevel(\
                    x,\
                    region,\
                    level,\
                    delta_level,\
                    min_new_spike,\
                    min_level,\
                    left,\
                    right\
                    ) :

    verbose = False
    
    #start at level: therefore reset level to level + delta_level
    level = level + delta_level

    #not done
    done = False
    
    #work var
    spike_width = left + right
    
    #work var
    pad = 10
    
    #empty list for return if needed
    spike_list = np.array([])

    #find spikes
    while not done and level > min_level :
        
        #update level
        level = level - delta_level

        #id: [x>level] and [region unoccupied] 
        spike_list = np.multiply(np.nonzero(x[:-spike_width-pad] > level)[0],\
                    1 - region[np.nonzero(x[:-spike_width-pad] > level)[0]])
        if verbose: print('[>level] and [not occupied]: ', spike_list)

        #first instance: x>level
        # spike_list = spike_list[ [3: nonzero [2: diff [1: prepend by 0], n=1] > 1] ]
        spike_list = spike_list[np.nonzero(np.diff(np.insert(spike_list, 0, 0), n=1, axis=0)>1)[0]]
        if verbose: print('[first instance >level] ',spike_list)

        #id: spike_list[max[x]] after first instance
        #spike_domain table: [spike_list index, L:spike_list, R:spike_list + spike_width]
        spike_domain = zip(np.arange(len(spike_list)), spike_list, spike_list + spike_width)
        #id valid: find max[x] in each domain * ( 1 - region validity: =1 if valid, <1 if invalid)
        for [n,L,R] in spike_domain :
            if verbose: print('[nLR]: ',n,L,R)
            spike_list[n] = (np.argmax(x[L:R]) + spike_list[n]) * (1 - np.sum(region[L:R]))

        if verbose: print('spike_list ',spike_list)
        #keep valid spikes: >0 index
        spike_list = spike_list[spike_list > 0]
        if verbose: print('spike_list ',spike_list)
        #spike_list > 0
        if len(spike_list > 0) :
            #keep valid spikes: spaced far enough apart
            spike_list = spike_list[\
                                    np.diff(\
                                        np.insert(spike_list,\
                                                  0,\
                                                  spike_list[0]-(right+1)),\
                                                  n=1,\
                                                  axis=0\
                                            ) > right\
                                    ]
            if verbose: print('spike_list ',spike_list)

        if len(spike_list) >= min_new_spike :
            done = True
            
        #level too low
        if level <= min_level :
        
            done = True
            
            level = -2.0
            
            print('warning: getNewLevel: went below min_level ', level)
            
    return level, spike_list


# ### getSpikeLevel

# In[ ]:n
#spike detect for given level
#GLOBAL: location, region
def getSpikeLevel(\
                    x,\
                    location,\
                    region,\
                    spike_index,\
                    level,\
                    left,\
                    right\
                    ) :
    
    #store level
    if len(spike_index) > 0 :
        location[spike_index] = level

        #mask region
        for this_spike in spike_index :
            region[this_spike - left : this_spike + right] = 1
        
    return location, region


# ### getCleanedUp
# In[ ]:
#clean region mask: spike forward and backward to remove edge effects
#GLOBAL: region
def getCleanedUp(\
                    level,\
                    x,\
                    region,\
                    left,\
                    right,\
                    ) :

    #work var
    
    #spike_width
    spike_width = left + right
    
    #half spike width
    half_spike_width = np.divmod(spike_width,2)[0]
    
    #mask spike too close
    #close_spike_list = (region[i] * region[i+spike_width])==1
    close_spike_list = np.nonzero(np.multiply(region[:-spike_width],region[spike_width:])==1)[0]
    for [a,b] in zip(close_spike_list, close_spike_list + spike_width) :
        region[a : b] = 1

    #mask: x > level not masked +- half_spike_width
    #[index list above level] * [region[index list above level]]
    above_level_list = np.multiply(np.nonzero(x > level)[0], 1 - region[np.nonzero(x > level)[0]])
    
    #keep nonzero index
    above_level_list = above_level_list[above_level_list > 0]
    
    #apply mask
    for [a,b] in zip(above_level_list - half_spike_width, above_level_list + half_spike_width) :
        region[a : b] = 1

    return region


# ### getRingingCleanedUp
# In[ ]:
#GLOBAL: location, region
def  getRingingCleanedUp(\
                         x,\
                         location,\
                         region,\
                         ring_cutoff,\
                         ring_threshold,\
                         ring_second,\
                         ring_num_period,\
                         mpd,\
                         level,\
                         mean_shift_n,\
                         left,\
                         right,\
                         neural_interval\
                         ) :
    
    verbose=False

    #ring_horizon: number of sample
    ring_horizon = np.int(ring_second/neural_interval + 0.5)

    #init location_list to location[level of interest]
    pad = 2 * ring_horizon
    location_list = np.nonzero(np.abs(location[:-pad] - level) < 0.0001)[0]

    #set length
    len_location_list = len(location_list)
    

    #exception: <= 10 spikes
    if len_location_list <= 2 :
        if verbose: print('Exception: getRingingCleanedup: length(location_list)<= 2')
        sys.exit()

    #look for rings
    for i in np.arange(len_location_list) :

        #select spike global index
        n = location_list[i]

        #get max value set
        #remove mean and scale by peak x[n]: use x[n-1] so detect x[n] as peak
        scale_x = (x[n-1 : n + ring_horizon]-np.mean(x[n-1:n+ring_horizon]))/x[n]
        [_, max_value_set] = detect_peaks(scale_x, mpd=mpd)
        #ratio over 50%
        max_value_set = max_value_set[np.nonzero(max_value_set > ring_cutoff)[0]]
        #over half of 10 periods yield a peak
        if len(max_value_set) > ring_num_period : 

            #compute ringing metric
            ring = np.sum(np.multiply(max_value_set, max_value_set))/len(max_value_set)

            #ring
            if ring > ring_threshold :

                if verbose: print('ring, x, max_value_set', n, ring, x[n], max_value_set)

                #ringing spike: remove spike: lose whole ring
                if verbose: print(np.nonzero(np.abs(location[n : n + ring_horizon] - level) < 0.0001)[0])
                location[n : n + ring_horizon] = 0.0

                #ringing: mask region
                region[n : n + ring_horizon] = 1



    #reset location_list = [location=level]
    location_list = np.nonzero(np.abs(location - level) < 0.0001)[0]
        
    #remove spike: relative peak x[n-L:n-L+mean_shift] height too hight
    #
    #zip spike_data: n=global index: [n , x_peak=x[n], L=n-L]
    spike_data = zip(location_list,\
                     x[location_list],\
                     location_list - left,\
                     location_list - left + mean_shift_n\
                     )
    for [n, x_peak, L, R] in spike_data :

        #reject spike: mean shift drove peak down too much
        mean_before_peak = np.mean(x[L:R])

        #reject spike if mean_before_peak too high
        #OR x_peak - mean before peak < 1.0 (bad for log(prom) later)
        if mean_before_peak > 0.75 * x_peak  or x_peak - mean_before_peak < 1.0 :
            if verbose: print('L, n, R, mean_before_peak, peak ', L, n, R, mean_before_peak, x_peak)
            #reject spike: leave region masked
            location[n] = 0.0
        
    #reset location_list = [location=level]
    location_list = np.nonzero( np.abs(location - level) < 0.0001 )[0]

    return  location, region, location_list

# ### getRidOfIsland
# In[ ]:
def getRidOfIsland(region, left, right) :

    verbose=False
    #do two things: (i) widen islands to spike_width, (ii) coalesce islands < spike_width apart
    
    #work var
    spike_width = left + right

    #
    #job (i): widen small island
    #
    if verbose: print('job(i): widen small island')
    #identify island[start:end] in region
    if verbose: print('region ', region)
    #island_detect: region[i+1]-region[i]
    detect_island = np.diff(region)
    if verbose: print('detect island ',detect_island)
    
    #start index: store region index
    start_of = np.nonzero(detect_island > 0)[0][:-1] + 1
    if verbose: print('start of ',start_of)
    
    #end index: not inclusive: store region index
    end_of = np.nonzero(detect_island < 0)[0][1:] + 1
    if verbose: print('end of ',end_of)
    
    if len(start_of) != len(end_of) :
        if verbose: print('Exception: getRidOfIsland: len(start_of) != len(end_of) ', len(start_of), len(end_of))
        sys.exit()

    #island_width: island indexing
    width_of = end_of - start_of
    if verbose: print('width_of ', width_of)
    
    #number_of_island
    num_island = len(start_of)
    if verbose: print('num islands ',num_island)

    #id small island: island indexing
    small_island = np.nonzero(width_of < spike_width)[0]
    if verbose: print('small_island ',small_island)
    
    #id small island widen: spike_width - small_island_width
    amount_to_widen = spike_width - width_of[small_island]
    if verbose: print('amount_to_widen ', amount_to_widen)
    
    #widen small island
    if len(small_island) > 0 :
        small_island_data = zip(start_of[small_island], amount_to_widen)
        for (start, widen) in small_island_data :
            if verbose: print('start-widen, start ',start-widen, start)
            region[start - widen : start] = 1

    if verbose: print('job(i) result: ', region)
    if verbose: print('job(ii): coalesce island too close')        
    #re_id island
    detect_island = np.diff(region)
    start_of = np.nonzero(detect_island > 0)[0][:-1] + 1
    end_of = np.nonzero(detect_island < 0)[0][1:] + 1
    if verbose: print('detect island ', detect_island)
    if verbose: print('start of ',start_of)
    if verbose: print('end of ',end_of)
     #
    #job (ii): coalesce islands separated by < spike_width
    #
    
    #find space between island
    space_between = start_of[1:] - end_of[:-1]
    if verbose: print('space_between ',space_between)
    
    #id close island: island indexing
    close_island = np.nonzero(space_between < spike_width)[0]
    if verbose: print('close island ', close_island)

    #coalesce island too close
    if len(close_island) > 0 :
        close_island_data = zip(end_of[close_island], start_of[close_island+1])
        for (end_of_previous, start_of_next) in close_island_data :
            if verbose: print('end_of_previous, start_of_next ', end_of_previous, start_of_next)
            region[end_of_previous : start_of_next] = 1

    return region


# ### verbose_plot: called from analysisLevelGetSpike
# In[ ]:
def plot_multiLevel1(x, plot_list, plot_level, plot_title) :
    # https://realpython.com/python-matplotlib-guide/
    # https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf
    fig, ax = plt.subplots(figsize=(16,3))
    ax.plot(x, color='blue', linewidth=1)
    ax.plot(region*10,  color='black', linewidth=1)
    ax.grid(axis='y')
    ax.scatter(plot_list,plot_level,color='darkred',marker='.',s=200)
    ax.set_title(plot_title)
    #ax.set_xlim(3120000, 3140000)
    #plt.savefig('foo.png')
    plt.show
    


# ### analysisLevelGetSpike
# In[ ]:
#spike detect
#GLOBAL: location, region

def analysisLevelGetSpike(\
                            x,\
                            x_getRingingCleanedUp,\
                            number_neural_sample,\
                            location,\
                            region,\
                            level,\
                            spike_list,\
                            ring_cutoff,\
                            ring_threshold,\
                            ring_second,\
                            ring_num_period,\
                            mpd,\
                            mean_shift_n,\
                            left,\
                            right,\
                            neural_interval\
                             ) :

    verbose_plot=False
    ##############################################################
    #[location, region] = [level at spike peak, mask spike region]
    ##############################################################
    print(' CALL: getSpikeLevel')
    [location, region] = getSpikeLevel(\
                                        x,\
                                        location,\
                                        region,\
                                        spike_list,\
                                        level,\
                                        left,\
                                        right\
                                        )

    print(' \t number of spike ',np.sum(location > 1.0e-05))
    #plot
    if verbose_plot :
        plot_list=np.nonzero(np.abs(location - level) < 0.0001)[0]
        plot_level = x[plot_list]
        plot_title = 'getSpikeLevel: reset LOCATION/REGION'
        plot_multiLevel1(x, plot_list=plot_list, plot_level=plot_level, plot_title=plot_title) 

    ############################
    #clean up spike region mask
    ############################
    print(' CALL: getCleanedUp')

    region = getCleanedUp(\
                            level,\
                            x,\
                            region,\
                            left,\
                            right\
                            )
    #plot
    if verbose_plot :
        plot_list = np.nonzero(np.abs(location - level) < 0.0001)[0]
        plot_level = x[plot_list]
        plot_title = 'getCLeanedUp: reset REGION'
        plot_multiLevel1(x, plot_list=plot_list, plot_level=plot_level, plot_title=plot_title) 
    #############################################
    #clean up ringing and construct location list
    #############################################
    print(' CALL: getRingingCleanedUp')

    [location, region, location_list] =\
        getRingingCleanedUp(\
                                x_getRingingCleanedUp,\
                                location,\
                                region,\
                                ring_cutoff,\
                                ring_threshold,\
                                ring_second,\
                                ring_num_period,\
                                mpd,\
                                level,\
                                mean_shift_n,\
                                left,\
                                right,\
                                neural_interval\
                                )

    #plot
    if verbose_plot :
        plot_list = location_list
        plot_level = x[plot_list]
        plot_title = 'getRingingCLeanedUp: reset LOCATION/REGION/LOCATION_LIST BUILT'
        plot_multiLevel1(x, plot_list=plot_list, plot_level=plot_level, plot_title=plot_title) 

    print(' \t after ringing cleaned up: number of spike ',np.sum(location > 1.0e-5))
    
    ####################################
    #get rid of island: edit region mask
    ####################################
    #1. widen anything not as wide as a spike to a single spike width
    #2. coalesce anything within 1 spike width
    print(' CALL: getRidOfIsland ')
    region = getRidOfIsland(region, left, right)
    
    #plot
    if verbose_plot :
        plot_list = location_list
        plot_level = x[plot_list]
        plot_title = 'getRidOfIsland: reset REGION'
        plot_multiLevel1(x, plot_list=plot_list, plot_level=plot_level, plot_title=plot_title) 

    return location, region, location_list


# ### plot results from multiLevel main
# In[ ]:

# https://realpython.com/python-matplotlib-guide/
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf
def plot_results(n_choice, xsmooth, region, location_list_plus_total, location_list_minus_total) :
    n_choice = 357722376 
    lower = n_choice
    upper = n_choice+1000
    xplot = xsmooth[lower:upper]
    regionplot = region[lower:upper]
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(xplot, color='blue', linewidth=1)
    ax.plot(regionplot*10,  color='black', linewidth=1)
    ax.grid(axis='y')
    plot_list = location_list_plus_total
    plot_list = plot_list[np.nonzero(plot_list < upper)[0]]
    plot_list = plot_list[np.nonzero(plot_list > lower)[0]]
    plot_level = xsmooth[plot_list]
    plot_list = plot_list - lower
    ax.scatter(plot_list,plot_level,color='darkgreen',marker='.',s=100)
    plot_list = location_list_minus_total
    plot_list = plot_list[np.nonzero(plot_list < upper)[0]]
    plot_list = plot_list[np.nonzero(plot_list > lower)[0]]
    plot_level = xsmooth[plot_list]
    plot_list = plot_list - lower
    ax.scatter(plot_list,plot_level,color='darkred',marker='.',s=100)
    #ax.set_xlim(3120000, 3140000)
    #plt.savefig('foo.png')
    plt.show
    
    return


# ## MultiLevel1 Main
# In[ ]:

def MultiLevel1(matfile = 'neural.mat',\
                outputspikefile = 'outputspike.csv',\
                metadatafile = 'metadata_MultiLevel1.txt',\
                smoothing_width = 4,\
                level_plus_factor = 0.8,\
                level_minus_factor = 0.9,\
                delta_level = 0.1,\
                min_level = 1.5,\
                min_new_spike_plus = 1000,\
                min_new_spike_minus = 500,\
                left = 20,\
                right = 100,\
                ring_threshold = 0.5,\
                ring_cutoff = 0.5,\
                ring_second = 0.06,\
                ring_num_period = 5,\
                mpd = 60,\
                mean_shift_n = 5):
    
    
    try:
        
        #load data
        [channel_number, xraw, neural_start, neural_interval,\
         lvpraw, lvp_start, lvp_interval, respraw,\
         resp_start, resp_interval] = read_Matlab_data(matfile)
    
        if neural_start <= 1.0e-10 :
            print('Exception: MultiLevel1 Main: zero neural start time')
            sys.exit()
    
        #test data: need to set
        #neural_interval = 0.00005
    
        #up to 4 is okay for speed
        #smoothing_width = 4
        xsmooth = gaussian_filter1d(xraw, smoothing_width)
    
        #requires memory: but is much faster
        #smooth input: getRingingCleanedUp smoothing
        #xsmooth = gaussian_filter1d(x, 20)
        #equivalence: convolution of 4 moving average filter
        x_getRingingCleanedUp = ourSmoothData(xsmooth, 4)
        x_getRingingCleanedUp = ourSmoothData(x_getRingingCleanedUp, 3)
        x_getRingingCleanedUp = ourSmoothData(x_getRingingCleanedUp, 2)
        x_getRingingCleanedUp = ourSmoothData(x_getRingingCleanedUp, 1)
    
        #testing on real data: take data chunk
        #xsmooth = xsmooth[395500000:395550000]
    
        #number of samples
        number_neural_sample = len(xsmooth)
    
        ######################
        #START: GLOBAL VARS
        ######################
        #set spike locations globally: location = level
        location = np.zeros(number_neural_sample)
    
        #region: mask spike regions: mask=1 is occupied, mask=0 unoccupied
        region = np.zeros(number_neural_sample).astype(int)
        #avoid ends: use around 50 on test data
        pad = 500
        region[:pad] = 1
        region[-pad:] = 1
    
        #init level plus slightly *below* +max to switch on finding
        #plus value immediately and allow smooth movement to more plus spike
        level_plus = level_plus_factor * np.max(xsmooth)
        #level_plus = 18.0
        print(' level_plus = ',level_plus)
    
        #init level minus slightly *below* -min to switch off finding
        level_minus = level_minus_factor * level_plus
        #level_minus = 17.0
        print(' level_minus = ',level_minus)
    
        #change in level
        #delta_level = 0.1
    
        #min level
        #min_level = 3.0
        #min_level = 15.0
    
        #min number of plus spike
        #min_new_spike_plus = 5
        #min_new_spike_plus = 1000
    
        #min number of plus spike
        #min_new_spike_minus = 3
        #min_new_spike_minus = 500
    
        #number of levels
        number_of_levels = 0
    
        ######################
        #END: GLOBAL VAR
        ######################
    
        ########################
        #START: GLOBAL CONSTANTS
        ########################
    
        #number of samples before spike peak
        #left = 20
    
        #number of samples at, and beyond, spike peak
        #right = 100
    
        #ring_threshold = ringing scale factor, i.e. ring ~1 or below
        #ring_threshold is a function of ring_second
        #default ~ 10*spike_width gives ring_threshold ~ 0.3
        #ring_threshold = 0.5
    
        #ring_cutoff
        #ring_cutoff = 0.5
    
        #default ring_second = 0.06sec: assumed to be 10*spike_width
        #without recovery: ~1/2 spike width = 0.003
        #ring_second = 0.06
    
        #ring number of period
        #ring_num_period = 5
    
        #peak detect: min peak distance
        #mpd=60
    
        #default mean_shift_n = 5
        #mean_shift_n = 5
    
        #not done
        done_multiLevel1 = False
    
        #placeholder
        location_list_plus_total = np.array([]).astype(int)
        location_list_minus_total = np.array([]).astype(int)
    
        #spike detect
        while not done_multiLevel1 :
    
            #increment number_of_levels
            number_of_levels = number_of_levels + 1
    
            #decide which to do next
            #get new plus level
            if level_plus > 0 :
                [level_plus, proposed_plus_index] =\
                    getNewLevel(\
                                    xsmooth,\
                                    region,\
                                    level_plus,\
                                    delta_level,\
                                    min_new_spike_plus,\
                                    min_level,\
                                    left,\
                                    right\
                                    )
                print(' CALL: getNewLevel: plus: number proposed ',len(proposed_plus_index))
    
            #get new minus level
            if level_minus > 0 :
                [level_minus, proposed_minus_index] =\
                    getNewLevel(\
                                    -xsmooth,\
                                    region,\
                                    level_minus,\
                                    delta_level,\
                                    min_new_spike_minus,\
                                    min_level,\
                                    left,\
                                    right\
                                    )
                print(' CALL: getNewLevel: minus: number proposed ',len(proposed_minus_index))
    
            print(' [level_plus, level_minus] = ',level_plus, level_minus)
    
            #plus case (careful that level_plus > 0): take plus if equal
            if level_plus >= level_minus and level_plus > 0 :
    
                print(' ****************************')
                print(' Getting Level Plus Spike', level_plus)
                print(' ****************************')
    
                #get level plus spike
                [ \
                 location,\
                 region,\
                 location_list_plus\
                 ] =\
                    analysisLevelGetSpike(\
                                          xsmooth,\
                                          x_getRingingCleanedUp,\
                                          number_neural_sample,\
                                          location,\
                                          region,\
                                          level_plus,\
                                          proposed_plus_index,\
                                          ring_cutoff,\
                                          ring_threshold,\
                                          ring_second,\
                                          ring_num_period,\
                                          mpd,\
                                          mean_shift_n,\
                                          left,\
                                          right,\
                                          neural_interval\
                                          )
    
                #concat new result: row vector: axis=0 default
                location_list_plus_total = np.concatenate([location_list_plus_total, location_list_plus])
    
            #minus case (careful that level_minus > 0)
            elif level_minus > level_plus and level_minus > 0 :
    
                print(' ****************************')
                print(' Getting Level Minus Spike', level_minus)
                print(' ****************************')
    
                #get level minus spike
                #flip sign of x values inputted
                [\
                 location,\
                 region,\
                 location_list_minus\
                 ] =\
                    analysisLevelGetSpike(\
                                          -xsmooth,\
                                          -x_getRingingCleanedUp,\
                                          number_neural_sample,\
                                          location,\
                                          region,\
                                          level_minus,\
                                          proposed_minus_index,\
                                          ring_cutoff,\
                                          ring_threshold,\
                                          ring_second,\
                                          ring_num_period,\
                                          mpd,\
                                          mean_shift_n,\
                                          left,\
                                          right,\
                                          neural_interval\
                                          )
    
                #concat new result: row vector: axis=0 default
                location_list_minus_total = np.concatenate([location_list_minus_total, location_list_minus])
    
            #no level available: done
            else :
                done_multiLevel1 = True
    
            print(' *********************************')
            print(' ******MultiLevel1 COMPLETE****', number_of_levels)
            print(' *********************************')
        write_MultiLevel1_spike(location_list_plus_total, location_list_minus_total,outputspikefile)
    
        write_MultiLevel1_metadata(metadatafile,\
                                  channel_number,\
                                  smoothing_width,\
                                  pad,\
                                  level_plus_factor,\
                                  level_minus_factor,\
                                  delta_level,\
                                  min_level,\
                                  min_new_spike_plus,\
                                  min_new_spike_minus,\
                                  left,\
                                  right,\
                                  ring_threshold,\
                                  ring_cutoff,\
                                  ring_second,\
                                  ring_num_period,\
                                  mpd,\
                                  mean_shift_n,\
                                  neural_start = neural_start,\
                                  neural_interval = neural_interval,\
                                  lvp_start = lvp_start,\
                                  lvp_interval = lvp_interval,\
                                  resp_start = resp_start,\
                                  resp_interval = resp_interval\
                                  )
        print('ML1 Done')
        return 
    except Exception as e:
        print('Error MultiLevel1:\n')
        print(e)

    # In[ ]:


    #n_choice = 357722376 
    #plot_results(n_choice, xsmooth, region, location_list_plus_total, location_list_minus_total) 


    # In[ ]:




