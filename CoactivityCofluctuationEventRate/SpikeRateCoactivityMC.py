#!/usr/bin/env python
# coding: utf-8

"""
Created on Fri Nov 27 16:42:35 2020

@author: Guy Kember, Alex Karavos, Koustubh Sudarshan
"""
# In[]:
# IMPORT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# In[]:
# FUNCTION: PLOT STATS

def plot_coactivity_stats(
                xlabel="",
                ylabel="",
                title="",
                figfilename="",
                x=[],\
                y=[],\
                ylim_min=0.0,\
                ylim_max=10.0,\
                comment_df=""
                ) :

    # init plot
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(x, y)
    
    # set y limits
    ax.set_ylim(ylim_min, ylim_max)

    #title
    ax.set_title(title)

    # set x,y labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # try to build comment labels
    try:

        #comment_df = pd.read_csv(comment_file)
        inter_times = comment_df['times'].to_list()
        inter_labels = comment_df['labels'].to_list()

        # build 2nd set of x labels
        # https://pythonmatplotlibtips.blogspot.com/2018/01/add-second-x-axis-below-first-x-axis-python-matplotlib-pyplot.html  
        # Set scond x-axis
        ax2 = ax.twiny()

        # build labels
        # intervention_label = ['A','B','C'] # xticklabels
        # intervention_time = np.array([2.0, 4.0, 6.0]) 
        # time (hours) is position of xticklabels on old x-axis
        ax2.set_xticks(inter_times)
        ax2.set_xticklabels(inter_labels,rotation=45,fontsize=8)
        #ax2.set_xlabel('Intervention')
        ax2.set_xlim(xmin=np.min(x),xmax=np.max(x))
        
    except:
        print('No Comment File')

    plt.title(title)
    plt.savefig(figfilename, bbox_inches='tight', pad_inches=0.02)
    np.savetxt(figfilename.replace('.pdf', '.csv'), y)

    #plt.show()
    plt.close()
    return 

# In[]:
# FUNCTION: PLOT COACTIVITY

def plot_results(
    xlabel="",
    ylabel="",
    title="",
    figfilename="",
    animal_name="",
    auto_corr=[],
    sample_index=[],
    sample_time=[],
    max_lag=-9,
    ytick_downsample=0,
    ytick_precision=0,
    dt=0.0,
    comment_df=""
):
    
    # strip nan
    auto_corr=np.nan_to_num(auto_corr, posinf=0.0, neginf=0.0)
    # attention metric: RELATIVE: DOUBLE sided hard threshold

    # set hard threshold
    #hard_threshold_plus = 0.75
    #hard_threshold_minus = 0.75

    # hard threshold
    #auto_corr[np.where(np.logical_and(auto_corr >= -hard_threshold_minus,auto_corr <= hard_threshold_plus))] = 0.0
    
    # set -10.0 to nan: places white line in graphic
    auto_corr[auto_corr==-999.9]=np.nan
        
    # init figure
    _, ax = plt.subplots(figsize=(20, np.int(20)/ytick_downsample))

    # make image
    pos = ax.imshow(np.flipud(auto_corr.transpose()), aspect="auto",vmin=-1, vmax=1)#, interpolation="hanning")

    # title
    ax.set_title(title)

    # ticks

    # x first' +'
    # xtick locn[wrt index imaged array specific dimension used] and label
    xtick_locn = np.linspace(0, len(sample_index), 16).astype(int)
    xtick_label = np.round(np.linspace(sample_time[0], np.max(sample_time), 16) / 3600, 3)    

    # place locn and label
    ax.set_xticks(xtick_locn)
    ax.set_xticklabels(xtick_label)

    # y second
    # ytick locn[wrt index imaged array specific dimension used] and label
    ytick_locn = np.linspace(max_lag-1, 0, np.int(max_lag / ytick_downsample)+1).astype(int)
    ytick_label = np.round(np.array(np.linspace(0, max_lag-1, np.int(max_lag / ytick_downsample) + 1)),ytick_precision)
  
    # place locn and label
    ax.set_yticks(ytick_locn)
    ax.set_yticklabels(ytick_label)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    #plt.ylabel("Period (sec)")
    #plt.tight_layout()
    # colorbar: https://matplotlib.org/3.1.0/gallery/color/colorbar_basics.html
    # add the colorbar using the figure's method,
    # telling which mappable [pos] we're talking about and
    # which axes object [ax] it should be near
    plt.colorbar(pos, ax=ax)
    # try to build comment labels
    try:
        #comment_df = pd.read_csv(comment_file)
        inter_times = comment_df['times'].to_list()
        inter_labels = comment_df['labels'].to_list()
        # build 2nd set of x labels
        # https://pythonmatplotlibtips.blogspot.com/2018/01/add-second-x-axis-below-first-x-axis-python-matplotlib-pyplot.html  
        # Set scond x-axis
        ax2 = ax.twiny()
        #build labels
#        intervention_label = ['A','B','C'] # xticklabels
#        intervention_time = np.array([2.0, 4.0, 6.0]) # time (hours) is position of xticklabels on old x-axis
        ax2.set_xticks(inter_times)
        ax2.set_xticklabels(inter_labels,rotation=45,fontsize=8)
        #ax2.set_xlabel('Intervention')
        ax2.set_xlim(xmin=np.min(xtick_label),xmax=np.max(xtick_label))
        
    except:
        print('No Comment File')

    plt.title(title)
    plt.savefig(figfilename, bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    plt.close()

    return

# In[]:
# compute sliding coactivity
def getSpikeRateCoactivity(
                        df="",
                        time="",
                        window=201,
                        ch_numbers="",
                        corr_threshold=0.75,
                        outdir=""
                        ):

    # set num channels
    num_chan = ch_numbers.shape[0] 
    
    # set half_win
    # calculate half win
    half_win = int((window - 1) / 2)
    
    # ensure oddness of window
    window = 2 * half_win + 1
        
    # init coactivity values list
    coactivity_vals = []
        
    # choose super-diagonal
    for super_diagonal in range(1,num_chan):
    
        # loop through available channels on super-diagonal
        for index in range(0, num_chan - super_diagonal):
     
            #set channel pair: names start at nonzero: indexes start at 0
            phys_name0 = ch_numbers[index]
            phys_name1 = ch_numbers[index+super_diagonal]
            name0 = "X_icn" + str(phys_name0)
            name1 = "X_icn" + str(phys_name1)
            #print(f"super {super_diagonal} index {index} name0 {phys_name0} name1 {phys_name1}")
            
            if (name0 in df.columns) and (name1 in df.columns):
                values = np.array(
                    df[name0].rolling(window=window, center=True).corr(df[name1])
                )[half_win:-half_win]
                # append values
                coactivity_vals.append(values)
            else:
                # put on a single zero
                coactivity_vals.append(np.zeros(1))
                
        # id end of super_diagonal: values will always exist
        #bug: coactivity_vals.append(np.ones(values.shape[0])*-999.9)
        coactivity_vals.append(np.ones(df.shape[0]-window + 1)*-999.9)

        
    # make list of array rectangular for numpy ndarray type
    # length one zero arrays to have length of last values
    for i in range(len(coactivity_vals)):
        if len(coactivity_vals[i])==1:
            #bug: coactivity_vals[i] = np.zeros(values.shape[0])
            coactivity_vals[i] = np.zeros(df.shape[0]-window + 1)
            
    # make into np ndarray
    coactivity_vals=np.array(coactivity_vals)

    # transpose for plotting
    coactivity_vals=np.transpose(coactivity_vals)
        
    # coactivity_vals.shape = [times, correlations]
    [rows, cols] = coactivity_vals.shape
    
    # warning: if nan members
    if len(np.nonzero(np.isnan(coactivity_vals))[0]) > 0:
        output_file = outdir + '/' + 'coact_WARNING.txt'
        f=open(output_file,'a+')
        f.write('encountered ' + str(len(np.nonzero(np.isnan(coactivity_vals))[0])) + ' invalid values')
        f.close()
    
    # count values above corr_threshold
    coactivity_stats=np.zeros(rows)
    # can be shortened: but easier to read this way: very little extra time
    for row in np.arange(rows):
        # remove nan
        ind1 = np.nonzero(~np.isnan(coactivity_vals[row]))[0]
        # remove -999.0
        ind2 = np.nonzero(np.abs(coactivity_vals[row][ind1])<=1.0)[0]
        # make filter_row
        filter_row = coactivity_vals[row][ind1]
        filter_row = filter_row[ind2]
        # remove zeros
        filter_row = filter_row[np.nonzero(np.abs(filter_row>0.01))[0]]
        if len(filter_row) > 0:
            # count values beyond desired value
            coactivity_stats[row] = len(np.nonzero(np.abs(filter_row)>corr_threshold)[0])
            # construct percentage
            coactivity_stats[row] = coactivity_stats[row] * 100.0 / len(filter_row)

    # retain correct times
    # centered
    #time=time[half_win:-half_win]
    # causal
    time=time[2*half_win:]
    

    return window, time, coactivity_vals, coactivity_stats

# In[]:
# compute gtime: global timestamp, grates: global rates
# store in DataFrame
def getDataFrame(filepaths="", ch_list="", rolling_mean_win=201, name="rate", outfile='dummy'):

    # open outputfile
    f = open(outfile,"a+")
    f.write('\n name is \n')
    f.write(name)    
    # set column names to actual icn#
    col_names = ["X_icn" + str(ch_list[i]) for i in range(len(filepaths))]    

    # init times and rates
    times = []
    rates = []
    
    for file in filepaths:
        df = pd.read_csv(file)
        rates.append(np.array(df[" rate"]))
        times.append(np.array(df["time"]))
    
    # In[]:
    # SINGLE TIMESTAMP
    
    # get equal rate arrays and decide on single timestamp series
    
    # find channel with earliest start time
    start_time = np.min([time[0] for time in times])
    
    # find channel with latest start time
    end_time = np.max([time[-1] for time in times])
    
    # set dt
    dt = times[0][1]-times[0][0]
    
    # convert to exact float 7 sig figs
    dt = int(dt*pow(10,7))/pow(10,7)
    
    #print('start end times  are ',start_time, end_time)
    
    # number of timestamps: slightly beyond max
    n_time = int((end_time-start_time)/dt + 0.5) + 3
    
    # set gtime: global timestamp
    gtime = start_time + np.arange(n_time)*dt
    
    # reset end time to last value
    end_time = gtime[-1]
    
    # In[]
    # make all the same length and pre/app-end first/last rate
    # they all share the same dt as gtime
    for i, [t, r] in enumerate(zip(times,rates)):
        # pad end with last entry
        # append
        n_append = int(np.abs(t[-1] - end_time)/dt)
        rates[i] = np.append(r, np.ones(n_append)*r[-1])
        times[i] = np.append(t, t[-1]+(np.arange(n_append)+1)*dt)
        # reset pointers for prepend
        t=times[i]
        r=rates[i]
        # prepend
        n_prepend = len(gtime) - len(t)
        rates[i] = np.insert(r, 0, np.ones(n_prepend)*r[0])
        times[i] = np.insert(t, 0, np.flip(t[0]-(np.arange(n_prepend)+1)*dt))
    
    
    # In[]
    # keep gtime inside bounds of times associated with rates for scipy interp1d
    max_start_time = np.max([time[0] for time in times])
    min_end_time = np.min([time[-1] for time in times])
    gtime = gtime[np.nonzero(np.logical_and(gtime>max_start_time,gtime<min_end_time))[0]]
    
    # In[]:
    #set grates: global rates, associated with gtime: global timestamp
    grates=[]
    for t, r in zip(times,rates):
        linear_interp = interp1d(t, r)
        grates.append(linear_interp(gtime))
    
    grates=np.array(grates)
    
    # In[]:
    # could embed in last loop but not good to entangle
    # at this time pandas only handle 1D array. Use strided to doing nD but
    # pointless for us - this is fast enough.
    # compute half win
    #rolling_mean_win = coactivity window
    # get half_win
    half_win = np.round((rolling_mean_win-1)/2).astype(int)
    # ensure center: may shift provided rolling_mean_win if user messed up
    # but a very small change
    rolling_mean_win = 2 * half_win + 1
    # rolling mean and pad
    
    rolling_mean = []
    for _, rates in enumerate(grates):
        if name=='rate':
            # Spikerate analysis
            rolling_grates = np.array(pd.Series(rates).rolling(rolling_mean_win).mean())
            mask=np.nonzero(np.isnan(rolling_grates))[0]
            rolling_grates[mask] = np.array(rates)[mask]
            rolling_mean.append(rolling_grates)
        else:
            # Spikestd analysis
            rolling_grates = np.array(pd.Series(rates).rolling(rolling_mean_win).std())
            mask=np.nonzero(np.isnan(rolling_grates))[0]
            rolling_grates[mask] = 0.0
            rolling_mean.append(rolling_grates)

    grates = np.array(rolling_mean)
    
    # In[]:
    # make dictionary of grates: global rates
    data = dict(zip(col_names, grates))
    
    # close outfile
    f.close()
    
    return [gtime, pd.DataFrame(data), dt]

def write_spikeratecoact_metadata(outfile='dummy',
                                  name='',
                                  animal_name='',
                                  window=-9,
                                  ytick_precision=-9,
                                  ytick_downsample=-9,
                                  filepaths=[],
                                  ch_list=[],
                                  ch_numbers=[],
                                  before_buffer=-9.0,
                                  after_buffer=-9.0
                                  ):
    
        # output metadata here -- isn't much of it
    f = open(outfile,'a+')

    f.write('task being performed \n')
    f.write(name)
    f.write('\n animal_name \n')
    f.write(animal_name)
    f.write('\n window \n')
    f.write(str(window))
    f.write('\n ytick_precision \n')
    f.write(str(ytick_precision))
    f.write('\n ytick_downsample \n')
    f.write(str(ytick_downsample))
    f.write('\n before buffer \n')
    f.write(str(before_buffer))
    f.write('\n after buffer \n')
    f.write(str(after_buffer))
    f.write('\n filepaths \n')

    for i, path in enumerate(filepaths):
        f.write(filepaths[i])
        f.write('\n')

    for i, path in enumerate(ch_list):
        f.write(str(ch_list[i]))
        f.write(' ')

    for i, path in enumerate(ch_numbers):
        f.write(str(ch_numbers[i]))
        f.write(' ')

    f.close()

    return


def SpikeRateCoact(animal_name="",\
                   filepaths="",\
                   ch_list="",\
                   ch_numbers="",\
                   comment_df="",\
                   outdir="",\
                   window=201,\
                   ytick_precision=0,\
                   ytick_downsample=2,
                   name='rate',
                   before_buffer=-9.0,
                   after_buffer=-9.0,
                   corr_threshold=0.75,
                   ylim_min=0.0,
                   ylim_max=10.0):

    if name != 'rate' and name != 'std':
        print('SpikeRateCoact: WARNING: name = ',name)
        print('REVERTING to: rate')
        name = 'rate'
 
    # SpikeRateCoactivity metadata file
    SC_metadatafile = outdir + '/metadata_SC.txt'
    
    # output metadata here -- isn't much of it
    write_spikeratecoact_metadata(outfile=SC_metadatafile,
                                  name=name,
                                  animal_name=animal_name,
                                  window=window,
                                  ytick_precision=ytick_precision,
                                  ytick_downsample=ytick_downsample,
                                  filepaths=filepaths,
                                  ch_list=ch_list,
                                  ch_numbers=ch_numbers,
                                  before_buffer=before_buffer,
                                  after_buffer=after_buffer)
    
    # In[]:
    # get global times and global rates (in df columns)
    [gtime, df, dt] = getDataFrame( filepaths=filepaths, 
                                    ch_list=ch_list, 
                                    rolling_mean_win=window,
                                    name=name,
                                    outfile=SC_metadatafile)
    
    # In[]:
    # put in SpikeratecoactivityMC: MAIN: take from NeuralAutocorrMC event handler
    # get coactivity
    [window, sample_time, coactivity_vals, coactivity_stats] = getSpikeRateCoactivity(
                            df=df,
                            time=gtime,
                            window=window,
                            ch_numbers=ch_numbers,
                            corr_threshold=corr_threshold,
                            outdir=outdir
                            )

    # plot rolling mean AllEvents
    xlabel = 'Time (hrs)'
    ylabel = 'Spike'+ name + 'Coactivity'
    event_name = 'AllEvents'
    title = animal_name + ' Spike_'+ event_name + name + ' Coactivity'
    figfilename = outdir + '/spike' + name +'_' + event_name + 'coact_' + animal_name + '.pdf'
    
    # use all indexes: used later for pruning
    sample_index=np.arange(len(sample_time))
    
    plot_results(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figfilename=figfilename,
        animal_name=animal_name,
        auto_corr=coactivity_vals,
        sample_index=sample_index,
        sample_time=sample_time,
        max_lag=coactivity_vals.shape[1],
        ytick_downsample=ytick_downsample,
        ytick_precision=ytick_precision,
        dt=dt,
        comment_df=comment_df
    )
    
    # plot coactivity_stats
    xlabel = 'Time (hrs)'
    ylabel = 'Percent above corr_threshold'
    event_name = 'AllEvents'
    title = animal_name + ' Spike_'+ event_name + name + ' Coactivity_Stats:' + 'percent > ' + str(corr_threshold )
    figfilename = outdir + '/spike' + name +'_' + event_name + 'coact_stats' + animal_name + '.pdf'
    
    plot_coactivity_stats(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figfilename=figfilename,
        x=sample_time/3600.0,\
        y=coactivity_stats,\
        ylim_min=ylim_min,\
        ylim_max=ylim_max,\
        comment_df=comment_df
        )

    #print(event_name + 'coactstats_' + animal_name + '.csv')
    #np.savetxt(event_name + 'coactstats_' + animal_name + '.csv', coactivity_stats, delimiter=",")
        
    #loop through start -> end interventions
    #comment_df = pd.read_csv(comment_file)
    f = open(SC_metadatafile, "a+")

    try:
        inter_times = np.array(comment_df['times'].to_list())*3600.0
        inter_labels = np.array(comment_df['labels'].to_list())
        inter_events = np.array(comment_df['events'].to_list())
        
        
        f.write('\n SAVE EVENT INFO \n')
        f.write('\n TIMES \n')
        np.savetxt(f, inter_times)
        f.write('\n LABELS \n')
        np.savetxt(f, inter_labels, fmt="%s")
        f.write('\n EVENTS \n')
        np.savetxt(f, inter_events, fmt="%s")
        f.flush()
              
        starts = np.array(np.nonzero(inter_events=='start')[0])
        ends = np.array(np.nonzero(inter_events=='end')[0])
        
        starts_times = inter_times[starts]
        starts_labels = inter_labels[starts]
        ends_times = inter_times[ends]
        
        f.write('\n START TIMES \n')
        np.savetxt(f, starts_times)
        f.write('\n START LABELS \n')
        np.savetxt(f, starts_labels, fmt="%s")
        f.write('\n END TIMES \n')
        np.savetxt(f, ends_times)
        f.flush()
        
        for start, end, event_name, event_num in zip(starts_times, ends_times, starts_labels, np.arange(len(starts_labels))):
            
            # update event_name to include the number of the event
            event_name = event_name + str(event_num)
            
            f.write('\n LOOP CHECK \n')
            f.write(str(start) + ' ')
            f.write(str(end) + ' ')
            f.write(event_name)
            f.write('\n')
            f.flush()

            # plot autocorr
            try:
                # global
                # extract start->end event
                # indices
                sub = np.nonzero(np.logical_and(sample_time>start-before_buffer, sample_time<end+after_buffer))[0]
                # indexes
                sample_indexsub=sample_index[sub]
                # times
                sample_timesub=sample_time[sub]
                # coactivity
                coactivitysub=coactivity_vals[sub,:]
                
                
                # plot coactivity around event
                title = animal_name + ' Spike_'+ event_name + 'Rate Coactivity'
                figfilename = outdir + '/spike' + name + '_' + event_name + 'coact_' + animal_name + '.pdf'
                   

                f.write('\n INDEX CHECK \n')
                f.write(str(sub.shape))
                f.flush()
                f.write(str(sample_indexsub.shape))
                f.flush()
                f.write(str(sample_timesub.shape))
                f.flush()
                f.write(str(coactivitysub.shape))
                f.flush()

                plot_results(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title,
                    figfilename=figfilename,
                    animal_name=animal_name,
                    auto_corr=coactivitysub,
                    sample_time=sample_timesub,
                    sample_index=sample_indexsub,
                    max_lag=coactivity_vals.shape[1],
                    ytick_downsample=ytick_downsample,
                    ytick_precision=ytick_precision,
                    dt=dt,
                    comment_df=comment_df
                )

                #
                # coactivity_stats plot
                #
                # coactivity_stats subset
                coactivity_statssub=coactivity_stats[sub]
 
                # plot coactivity_stats around event
                title = animal_name + ' Spike_'+ event_name + name + ' Coactivity_Stats:' + 'percent > ' + str(corr_threshold )
                figfilename = outdir + '/spike' + name + '_' + event_name + 'coact_stats' + animal_name + '.pdf'
                ylabel = 'Percent above corr_threshold'
                                
                plot_coactivity_stats(
                        xlabel=xlabel,
                        ylabel=ylabel,
                        title=title,
                        figfilename=figfilename,
                        x=sample_timesub/3600.0,\
                        y=coactivity_statssub,\
                        ylim_min=ylim_min,\
                        ylim_max=ylim_max,\
                        comment_df=comment_df
                )

                f.write('***ABLE TO PLOT EVENT***')
                f.write(event_name)
                f.write('\n')
                f.flush()
      
            except:
                f.write('***UNABLE TO PLOT EVENT***')
                f.write(event_name)
                f.write('\n')
                f.flush()


    except:
        print('Corrupt comment_df ')
        f.write('Corrupt comment_df ')
        f.flush()
    
    f.close()

    return
