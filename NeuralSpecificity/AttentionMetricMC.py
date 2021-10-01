#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
#import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import copy
from scipy.signal import savgol_filter
from scipy.stats import entropy
#from scipy import ndimage

# # SlidingHistogram: AttentionMetric
# ## read/write helper functions for initialization

"""
process output from MultiLevel2: create Attention Metric: LVP, RESP
"""
__author__ = "Koustubh Sudarshan, Alex Karavos, Guy Kember"
__version__ = "1.0"
__license__ = "Apache v2.0"

# ## read/write helper functions for initialization

# ### read_Matlab_data

# In[ ]:


# load data
# neural: never loaded
# load: either: lvp OR resp
# changed name to read_Matlab_target FROM read_Matlab_data since neural is never read
# U.keys() is top list: U['keyname'].keys: is member list
# ['comment', 'interval', 'length', 'offset', 'scale', 'start', 'title', 'units', 'values']
def read_Matlab_target(
    hasNEURAL=False,
    hasLVP=False,
    hasRESP=False,
    neural_file="neural.mat",
    lvp_file="lvp.mat",
    resp_file="resp.mat",
):

    try:
        if hasNEURAL:

            # load
            S = h5py.File("neural.mat", "r")

            # abstract key list
            S_list = list([])
            for this_key in S:
                S_list.append(this_key)

            # channel number
            channel_number = S_list[0]

            # extract neural values: flatten for later cat
            xraw = np.array(S.get(S_list[0])["values"]).flatten()

            # normalize xraw
            xraw = (xraw - np.mean(xraw)) / np.std(xraw)

            # create times
            # sampling interval
            neural_interval = S.get(S_list[0])["interval"][0][0]
            # neural start time
            neural_start = S.get(S_list[0])["start"][0][0]

            # close input file
            S.close()

        else:

            channel_number = "ICNunknown"
            xraw = np.array([])
            neural_start = -9.0
            neural_interval = -9.0

            print("Warning: Neural data not read: hasNeural=", hasNEURAL)

        if hasLVP:

            # load
            T = h5py.File(lvp_file, "r")

            # abstract list
            T_list = list([])
            for this_key in T:
                T_list.append(this_key)

            # extract lvp
            # lvp values: : flatten for later cat
            lvpraw = np.array(T.get(T_list[0])["values"]).flatten()

            # start time
            lvp_start = T.get(T_list[0])["start"][0][0]

            # lvp interval
            lvp_interval = T.get(T_list[0])["interval"][0][0]

            # close input file
            T.close()

        else:
            print("Warning: read_Matlab_data: hasLVP=", hasLVP)
            # return empty
            lvpraw = np.array([])
            lvp_start = -9.0
            lvp_interval = -9.0

        # resp
        if hasRESP:

            # load
            U = h5py.File(resp_file, "r")

            # abstract list
            U_list = list([])
            for this_key in U:
                U_list.append(this_key)

            # extract respiratory
            # values: : flatten for later cat
            respraw = np.array(U.get(U_list[0])["values"]).flatten()

            # start time
            resp_start = U.get(U_list[0])["start"][0][0]

            # resp interval
            resp_interval = U.get(U_list[0])["interval"][0][0]

            # close input file
            U.close()

        else:
            print("Warning: read_Matlab_data: hasRESP=", hasRESP)
            # return empty
            respraw = np.array([])
            resp_start = -9.0
            resp_interval = -9.0

        return (
            channel_number,
            xraw,
            neural_start,
            neural_interval,
            lvpraw,
            lvp_start,
            lvp_interval,
            respraw,
            resp_start,
            resp_interval,
        )

    except Exception as e:
        print(e)
        print("Exception: read_Matlab_data")
        sys.exit()


# ### read MultiLevel2 spike

# In[ ]:


# load multi channel results from MultiLevel2
# sort to ascending time
# df.columns = ['filename','channel']: access via df['filename']
# provide read_csv: names


def read_MultiLevel2_spike(file_dataframe):

    df = file_dataframe
    channel_list = df["channel"]

    # *.csv: vars: spike_time,spike,lvp,lvp_phase,resp,resp_phase,plus_minus_id,prom,width,spike_x_level
    spike_time = np.array([])
    # spike: never used
    lvp = np.array([])
    lvp_phase = np.array([]).astype(int)
    resp = np.array([])
    resp_phase = np.array([]).astype(int)
    plus_minus_id = np.array([]).astype(int)
    prom = np.array([])
    width = np.array([])
    spike_level = np.array([])
    # computed data
    dataChannel = np.array([]).astype(int)
    dataChannelIndex = np.array([]).astype(int)

    for (file, channel) in zip(df["filename"], df["channel"]):
        df = pd.read_csv(file)  # , usecols=['spike_time', 'lvp', 'spike_x_level'])
        spike_time = np.concatenate([spike_time, df["spike_time"]])
        lvp = np.concatenate([lvp, df["lvp"]])
        lvp_phase = np.concatenate([lvp_phase, df["lvp_phase"]])
        resp = np.concatenate([resp, df["resp"]])
        resp_phase = np.concatenate([resp_phase, df["resp_phase"]])
        plus_minus_id = np.concatenate([plus_minus_id, df["plus_minus_id"]])
        prom = np.concatenate([prom, df["prom"]])
        width = np.concatenate([width, df["width"]])
        spike_level = np.concatenate([spike_level, df["spike_x_level"]])
        # computed data
        dataChannel = np.concatenate(
            [dataChannel, channel * np.ones(len(df["spike_x_level"])).astype(int)]
        )
        dataChannelIndex = np.concatenate(
            [dataChannelIndex, np.arange(len(df["spike_x_level"])).astype(int)]
        )

    # sort: ascending sample_time: sample_time[not monotonic: sep channel]
    index = np.argsort(spike_time)

    return (
        channel_list,
        dataChannel[index],
        dataChannelIndex[index],
        spike_time[index],
        lvp[index],
        lvp_phase[index],
        resp[index],
        resp_phase[index],
        plus_minus_id[index],
        prom[index],
        width[index],
        spike_level[index],
    )


# ### write AttentionMetric metadata

# In[ ]:


def write_AttentionMetric_metadata(
    AM_metadatafile="",
    hasLVP=False,
    hasRESP=False,
    uncurated=False,
    curated=False,
    template=False,
    hard_threshold=0.5,
    level_plus=0.0,
    level_minus=0.0,
    window=600.0,
    down=100,
    num_bin=100,
    art_fraction=0.5,
    channel_name="",
    num_art_compare=5,
    tar_start=-9.0,
    tar_interval=-9.0,
    neural_interval=-9.0,
    tar_upp_trans=0.1,
    tar_low_trans=0.1,
    x_lower_lim=-9.0,
    x_upper_lim=-9.0,
    badtar_upper_lim=np.inf,
    badtar_lower_lim=-np.inf,
    hard_threshold_rel_to_mean=False,
    att_metric=False,
    before_buffer=-9.0,
    after_buffer=-9.0
):

    f = open(AM_metadatafile, "w")

    f.write("\n hasLVP \n")
    f.write(str(hasLVP))

    f.write("\n hasRESP \n")
    f.write(str(hasRESP))

    f.write("\n uncurated \n")
    f.write(str(uncurated))

    f.write("\n curated \n")
    f.write(str(curated))

    f.write("\n template \n")
    f.write(str(template))

    f.write("level_plus \n")
    f.write(str(level_plus))

    f.write("\n level_minus \n")
    f.write(str(level_minus))

    f.write("\n hard_threshold \n")
    f.write(str(hard_threshold))

    f.write("\n window \n")
    f.write(str(window))

    f.write("\n down \n")
    f.write(str(down))

    f.write("\n channel_name \n")
    f.write(str(channel_name))

    f.write("\n num_art_compare \n")
    f.write(str(num_art_compare))

    f.write("\n tar_start \n")
    f.write(str(tar_start))

    f.write("\n tar_interval \n")
    f.write(str(tar_interval))

    f.write("\n neural_interval \n")
    f.write(str(neural_interval))

    f.write("\n tar_upp_trans \n")
    f.write(str(tar_upp_trans))

    f.write("\n tar_low_trans \n")
    f.write(str(tar_low_trans))

    f.write("\n x_lower_lim \n")
    f.write(str(x_lower_lim))

    f.write("\n badtar_upper_lim \n")
    f.write(str(badtar_upper_lim))

    f.write("\n badtar_lower_lim \n")
    f.write(str(badtar_lower_lim))

    f.write("\n x_upper_lim \n")
    f.write(str(x_upper_lim))

    f.write("\n hard_threshold_rel_to_mean \n")
    f.write(str(hard_threshold_rel_to_mean))

    f.write("\n att_metric \n")
    f.write(str(att_metric))
    
    f.write('\n before buffer \n')
    f.write(str(before_buffer))

    f.write('\n after buffer \n')
    f.write(str(after_buffer))
    
    # close output file
    f.close()

    return


# ### remove artifact
#

# In[4]:


def remove_artifact(
    num_art_compare, channel_list, dataChannel, spike_time, art_threshold
):
    # diff_spike_time
    # diff_spike_time = np.diff(spike_time)
    # pass1: id diff[t1, t1+a, t1+b, t1+c, ...] < threshold: excluding t1
    # spike_time[np.nonzero(diff_spike_time < art_threshold)[0] + 1] = -9.0
    # pass2: id t1
    # spike_time[np.nonzero(diff_spike_time < art_threshold)[0]] = -9.0
    # up to five artifact for check
    n_channel = np.min([len(channel_list), num_art_compare])
    diff_index = np.nonzero(
        spike_time[n_channel:] - spike_time[:-n_channel] <= art_threshold
    )[0]
    print("\t \t remove_artifact: n_channel: num art: ", n_channel, len(diff_index))
    zipit = zip(diff_index, diff_index + n_channel)
    for L, R in zipit:
        spike_time[L:R] = -9.0
    # id artifact index
    index = np.nonzero(np.logical_and(spike_time > 0, dataChannel > 0))[0]

    return index


# plot results

# In[ ]:


def plot_results(
    channel_name,
    attention,
    attention_random_copy,
    hard_threshold,
    start_index,
    equal_spike_time,
    equal_spike_index,
    event_name="",    
    hasLVP=False,
    hasRESP=False,
    hasAttention=False,
    hasAttentionSample=False,
    hasAttentionRandom=False,
    tar_upp_trans=0.1,
    tar_low_trans=0.1,
    lower_bin=-9.0,
    upper_bin=-9.0,
    window=-9.0,
    att_metric=False,
    num_bin=100,
    outdir="",
    comment_df=""
):

    if hasLVP:
        target_name = "LVP"
    elif hasRESP:
        target_name = "RESP"
    else:
        print("Exception: graphic: bad choice LVP or RESP")
        sys.exit()

    if hasAttention:
        # update attention name to include possible event so can distinguish
        # pure attention from attention zoomed in around an event
        attention_name = "Attention_" + event_name + "Metric"
    elif hasAttentionSample:
        attention_name = "Attention_" + event_name + "Sample_Metric"
    elif hasAttentionRandom:
        attention_name = "Attention_" + event_name + "Random_Metric"
    else:
        print(
            "Exception: graphic: bad choice: attention or attention_sample or attention_random"
        )
        sys.exit()

    if lower_bin == -9.0 or upper_bin == -9.0:
        print("Exception: graphc: bad choice: lower_bin OR upper_bin")
        sys.exit()

    if window == -9.0:
        print("Exception: graphic: bad choice: window")
        sys.exit()

    if att_metric:
        att_metric_name = "relative"
    else:
        att_metric_name = "not_relative"

    # copy/smoothed attention
    hard_attention = np.copy(attention)

    #
    # https://stackoverflow.com/questions/37749900/how-to-disregard-the-nan-data-point-in-numpy-array-and-generate-the-normalized-d
    my_cmap = copy.copy(plt.cm.get_cmap("viridis"))  # get a copy of the gray color map
    my_cmap.set_bad(alpha=0)
    # transpose and flipud for imshow
    # case: extent = none: https://matplotlib.org/3.2.1/tutorials/intermediate/imshow_extent.html
    # hard filter (could use np.logical_or for one-line version of next 3 lines)
    # used for SINGLE hard threshold: NOT RELATIVE
    # hard_attention[hard_attention > hard_threshold] = hard_threshold
    # used for DOUBLE hard threshold: IS RELATIVE
    #
    if att_metric:
        # attention metric: RELATIVE: DOUBLE sided hard threshold
        hard_attention[hard_attention > hard_threshold] = 1.0
        hard_attention[np.abs(hard_attention) <= hard_threshold] = 0.0
        hard_attention[hard_attention < -hard_threshold] = -1.0
    else:
        # attention metric: NOT RELATIVE: SINGLE sided hard threshold: SATURATE: KEEP VALUE IF BELOW
        # shannon entropy has no meaning here
        hard_attention[hard_attention > hard_threshold] = hard_threshold

    # mask values where attention is beyond peak pressure location
    rows, _ = attention.shape
    for this_row in np.arange(rows):
        # find diastole edge
        nan_start = np.argmax(attention_random_copy[this_row, :] >= tar_upp_trans)
        if not nan_start == 0:
            hard_attention[this_row, 0:nan_start] = np.nan
        # find systole edge
        nan_start = np.argmax(
            np.flip(attention_random_copy[this_row, :]) >= tar_low_trans
        )
        if not nan_start == 0:
            hard_attention[this_row, -nan_start - 2 :] = np.nan

    # make attention_plot: for assignment
    copy_attention = np.ones([len(start_index), num_bin]) * -999.0
    # make shannon entropy array
    shannon_entropy = np.ones([len(start_index)]) * -999.0
    # map to unequally spaced -> equally spaced
    copy_attention[equal_spike_index, :] = hard_attention[start_index, :]
    # construct shannon entropy
    for this_row in np.arange(len(start_index)):
        if copy_attention[this_row, 0] != -999.0 and att_metric:
            # being careful
            copy_this_row = copy_attention[this_row, :]
            copy_this_row = copy_this_row[np.nonzero(~np.isnan(copy_this_row))[0]]
            if len(copy_this_row) > 4:
                # use abs(difference) entropy
                # maps [-1,0,1] to [0,1,2] giving range=[-0.5,2.5]
                not_nan = np.abs(np.diff(copy_this_row))
                p, _ = np.histogram(not_nan, 3, range=[-0.5,2.5], density=True)
                # 
                # use raw entropy: not used
                #not_nan = copy_this_row
                #p, _ = np.histogram(not_nan, 3, range=[-1.5,1.5], density=True)

                # compute entropy
                shannon_entropy[this_row] = entropy(p, base=3)
    # fill empty rows: zero row is ALWAYS filled
    for this_row in np.arange(len(start_index)-1)+1:
        # backfill if necessary
        if copy_attention[this_row, 0] == -999.0:
            copy_attention[this_row, :] = copy_attention[this_row-1, :]

    # output shannon entropy
    outputFilename = outdir + '/' + event_name + '_shannon_entropy.csv'
    df = pd.DataFrame(\
        {'time':equal_spike_time[::10],\
         'shannon_entropy':shannon_entropy[::10],\
         }\
         )
    # output results
    df.to_csv(outputFilename, index=False)

    # output to binary file h5py format
    #if hasLVP:
    #    hf = h5py.File(outdir + 'att_metric_LVP.h5', 'w')
    #    hf.create_dataset('att_metric', data=copy_attention)
    #    hf.create_dataset('spike_time', data=equal_spike_time + window/2.0)
    #    hf.close()

    # init figure
    fig, ax = plt.subplots(figsize=(20, 5))
    # make image
    pos = ax.imshow(
        np.flipud(copy_attention.transpose()),
        cmap=my_cmap,
        aspect="auto",
        interpolation="hanning",
    )
    # title
    ax.set_title(
          event_name
        + " "
        + channel_name
        + " "
        + attention_name
        + " "
        + att_metric_name
        + " "
        + target_name
        + " vs. Time, hard_threshold = "
        + str(hard_threshold)
    )
    # ticks
    # x first
    # xtick locn[wrt index imaged array specific dimension used] and label
    xtick_locn = np.linspace(0, len(equal_spike_time), 36).astype(int)
    xtick_label = np.round(
        np.linspace(
            equal_spike_time[0] + window,
            equal_spike_time[len(start_index) - 1] + window,
            36,
        )
        / 3600,
        2,
    )
    # place locn and label
    ax.set_xticks(xtick_locn)
    ax.set_xticklabels(xtick_label)
    # y second
    # ytick locn[wrt index imaged array specific dimension used] and label
    ytick_locn = np.linspace(num_bin, 0, 11).astype(int)
    ytick_label = np.round(np.linspace(lower_bin, upper_bin, 11),2) #.astype(int)
    # place locn and label
    ax.set_yticks(ytick_locn)
    ax.set_yticklabels(ytick_label)

    plt.xlabel("Time (hrs)")
    plt.ylabel(target_name)
    plt.tight_layout()   
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
    
    # save fig
    plt.savefig(
        outdir
        + "image_"
        + channel_name
        + "_"
        + attention_name
        + "_"
        + att_metric_name
        + "_"
        + target_name
        + "_"
        + str(np.int(hard_threshold * 100))
        + ".pdf", bbox_inches='tight', pad_inches=0.02
    )

    # plt.show()
    plt.close()
    return

# ## Attention Metric Main

# In[ ]:

###need to read neural start and interval from somewhere other than neural.mat


def AttentionMetric(
    file_df="",
    target_file="",
    comment_df="",
    AM_metadatafile="",
    outdir="",
    hasLVP=False,
    hasRESP=False,
    uncurated=False,
    curated=False,
    template=False,
    hard_threshold=0.5,
    level_plus=1.5,
    level_minus=1.5,
    window=20.0, #Default changes to 20s for the surrogate data used
    down=10,
    num_bin=200,
    art_fraction=0.5,
    channel_name="",
    num_art_compare=5,
    neural_interval=0.00005,
    tar_upp_trans=0.1,
    tar_low_trans=0.1,
    x_lower_lim=-9.0,
    x_upper_lim=-9.0,
    badtar_lower_lim=-np.inf,
    badtar_upper_lim=np.inf,
    hard_threshold_rel_to_mean=False,
    att_metric=True,
    before_buffer=-9.0,
    after_buffer=-9.0
):

    if np.sum([hasLVP, hasRESP]) != 1:
        print("Exception: choose LVP or RESP for input")
        sys.exit()

    if np.sum([uncurated, curated, template]) != 1:
        print("Exception: choose uncurated or curated or template")
        sys.exit()
    #
    # read Matlab target
    #
    # load multi channel results from MultiLevel2: sorted ascending spike_time
    # channel_list,dataChannel,dataChannelIndex,spike_time,
    # lvp,lvp_phase,resp,resp_phase,plus_minus_id,prom,width,spike_x_level
    #
    if hasLVP:
        # raw target
        [_, _, _, _, tarraw, tar_start, tar_interval, _, _, _] = read_Matlab_target(
            hasLVP=True, lvp_file=target_file
        )
        # spike data
        [
            channel_list,
            dataChannel,
            _,
            spike_time,
            tar,
            _,
            _,
            _,
            _,
            _,
            _,
            spike_level,
        ] = read_MultiLevel2_spike(file_dataframe=file_df)

    else:
        # raw target
        [_, _, _, _, _, _, _, tarraw, tar_start, tar_interval] = read_Matlab_target(
            hasRESP=True, resp_file=target_file
        )
        # spike data
        [
            channel_list,
            dataChannel,
            _,
            spike_time,
            _,
            _,
            tar,
            _,
            _,
            _,
            _,
            spike_level,
        ] = read_MultiLevel2_spike(file_dataframe=file_df)



    # select desired range
    print("select desired range")
    # keep time: > tar_start [time]
    # id index
    index = np.nonzero(spike_time > tar_start)[0]
    # keep beyond
    spike_time = spike_time[index]
    spike_level = spike_level[index]
    dataChannel = dataChannel[index]
    tar = tar[index]

    # num spike to be used
    num_spike_with_art = len(dataChannel) - np.sum(dataChannel == -9)

    # remove artifact
    #
    print("remove artifact")
    art_threshold = neural_interval * art_fraction
    # id index
    index = remove_artifact(
        num_art_compare, channel_list, dataChannel, spike_time, art_threshold
    )
    # remove artifact
    spike_time = spike_time[index]
    spike_level = spike_level[index]
    dataChannel = dataChannel[index]
    tar = tar[index]

    print("\t \t number: before,  after ", num_spike_with_art, len(spike_time))

    # choose spike range
    # id index
    # index = np.nonzero(np.logical_or(spike_level >= level_plus, spike_level <= -level_minus))[0]
    index = np.nonzero(
        np.logical_or(
            np.logical_and(spike_level >= level_plus, spike_level <= 500.0),
            np.logical_and(spike_level <= -level_minus, spike_level >= -500.0),
        )
    )[0]
        
    # choose level based on plus/minus relative difference
    # same when ~gaussian and stay above this level
    #spike_level = np.random.randn(10000)
    #bin_width = 0.1
    #plus_spikes = spike_level[spike_level>0]
    #minus_spikes = np.abs(spike_level[spike_level<0])
    #max_level = np.max([np.max(plus_spikes), np.max(minus_spikes)])
    #n = np.int(max_level/bin_width + 0.5)
    #bin_edges = np.linspace(0,max_level,n+1)
    #plus_hist, _ = np.histogram(plus_spikes, bin_edges) 
    #minus_hist, _ = np.histogram(minus_spikes, bin_edges)
    #m = np.argmax(np.divide(np.abs(plus_hist - minus_hist), 2*(plus_hist+minus_hist+1.0e-05))>0.05)
    #level_plus = np.min([bin_edges[m],2.0])
    #level_minus = level_plus
        
    # set spike_time, spike_level, dataChannel, tar
    spike_time = spike_time[index]
    spike_level = spike_level[index]
    dataChannel = dataChannel[index]
    tar = tar[index]

    # remove bad target
    #
    print("remove bad target")
    # id index
    # default: badtar_lower=-np.inf, badtar_upper=np.inf
    index = np.nonzero(
        np.logical_and(tar >= badtar_lower_lim, tar <= badtar_upper_lim)
    )[0]
    # remove bad target
    spike_time = spike_time[index]
    spike_level = spike_level[index]
    dataChannel = dataChannel[index]
    tar = tar[index]

    print("set time limits")
    # choose time bounds
    if x_lower_lim > 0.0 and x_upper_lim > 0.0:
        # id index
        index = np.nonzero(
            np.logical_and(spike_time >= x_lower_lim, spike_time <= x_upper_lim)
        )[0]

        # set spike_time, spike_level, dataChannel, tar
        spike_time = spike_time[index]
        spike_level = spike_level[index]
        dataChannel = dataChannel[index]
        tar = tar[index]
    else:
        x_lower_lim = np.min(spike_time)
        x_upper_lim = np.max(spike_time)


    #
    # dump metadata
    write_AttentionMetric_metadata(
        AM_metadatafile=AM_metadatafile,
        hasLVP=hasLVP,
        hasRESP=hasRESP,
        uncurated=uncurated,
        curated=curated,
        template=template,
        hard_threshold=hard_threshold,
        level_plus=level_plus,
        level_minus=level_minus,
        window=window,
        down=down,
        num_bin=num_bin,
        art_fraction=art_fraction,
        channel_name=channel_name,
        num_art_compare=num_art_compare,
        tar_start=tar_start,
        tar_interval=tar_interval,
        neural_interval=neural_interval,
        tar_upp_trans=tar_upp_trans,
        tar_low_trans=tar_low_trans,
        x_lower_lim=x_lower_lim,
        x_upper_lim=x_upper_lim,
        badtar_upper_lim=badtar_upper_lim,
        badtar_lower_lim=badtar_lower_lim,
        hard_threshold_rel_to_mean=hard_threshold_rel_to_mean,
        att_metric=att_metric,
        before_buffer=before_buffer,
        after_buffer=after_buffer
    )        
        
        

    # sliding histogram
    #
    print("set [sliding] histogram parameters")
    # bin limit: lower_bin[leftmost edge] ... upper_bin[righmost edge]
    lower_bin = np.min(tar)
    upper_bin = np.max(tar)

    print("attention [sliding] histogram")
    print("\t start index: end index: attention [sliding] histogram")
    # end index window: ref from spike_time[0...]
    # note: compute spike_times = spike_time + window once: use memory: save spike_time
    # note: stay 1.1 * window from end for safety
    spike_times = (
        spike_time[: np.argmax(spike_time + 1.1 * window > np.max(spike_time))] + window
    )

    # list comprehension: slow code: for loop
    end_index = [np.argmax(spike_time > aspike_time) for aspike_time in spike_times]
    # make np
    end_index = np.array(end_index).astype(int)
    # prune zero
    len_end_index = len(end_index)
    end_index = end_index[end_index > 0]
    # warning: this may cause problem: argmax returns 0 if index cannot be found
    if len(end_index) - len_end_index != 0:
        print("WARNING: pruned zero entries")

    # make start_index: very important vector
    start_index = np.arange(len(end_index)).astype(int)

    # exception
    if np.min(end_index - start_index) <= 0:
        print("Exception: end_index <= start_index")
        sys.exit()

    print("\t init attention [sliding] histogram")
    # init: attention [sliding] histogram
    attention = np.zeros([len(start_index), num_bin])
    
    # MAJOR CHANGE: 
    # 1. best to set this early so that the code only uses spike time
    # at beginning of the window -> end of window and then shift time forward
    # half the window in the plot routine
    # 2. best to ensure that spike_time is the SAME SIZE as ATTENTION
    # 3. means that we do not need spike_time[start] below which simplifies
    # the data structure
    
    spike_time = spike_time[start_index]

    # attention_random_not_set init to 0 [True]: little weird to do this way: downsample requires
    attention_not_set = np.zeros(attention.shape[0]).astype(int)

    # attention [sliding] histogram
    print("\t find attention [sliding] histogram")
    for (start, end) in zip(start_index, end_index):
        [attention[start, :], _] = np.histogram(
            tar[start:end], num_bin, range=(lower_bin, upper_bin), density=True
        )
        #reduce noise level due to variable neural sampling rate
        #this is NOT done for random since that is full data and is exact
        if ~np.any(np.isnan(attention[start,:])):
            attention[start,:] = savgol_filter(attention[start,:],11,3)
        attention_not_set[start] = 1

    # fill empty location: attention random: due to downsampling: speed
    print("\t fill empty attention locations")
    attention_not_set = np.nonzero(attention_not_set == 0)[0]
    for index_not_set in attention_not_set:
        print("do reset")
        attention[index_not_set, :] = attention[index_not_set - 1, :]

    # attention random/sample [sliding] histogram
    print("attention random [sliding] histogram")
    print("\t start index: end index: attention_random [sliding] histogram")
    tarraw_start_index = np.array(
        (spike_time[start_index] - tar_start) / tar_interval + 0.5
    ).astype(int)
    # end index should be okay: stay inside 2 * window for spike_time[start_index]
    tarraw_end_index = tarraw_start_index + np.int(window / tar_interval + 0.5)
    # init: attention random/sample [sliding] histogram
    print("\t init attention random [sliding] histogram")
    attention_random = np.zeros([len(start_index), num_bin])
    attention_sample = np.zeros([len(start_index), num_bin])
    # attention_random_not_set init to 0 [True]: little weird to do this way: downsample requires
    attention_random_not_set = np.zeros(attention_random.shape[0]).astype(int)
    # set work var
    len_zip = len(tarraw_start_index)

    for (i, start, end, tarstart, tarend) in zip(
        np.arange(len(start_index[0:len_zip:down])),
        start_index[0:len_zip:down],
        end_index[0:len_zip:down],
        tarraw_start_index[0:len_zip:down],
        tarraw_end_index[0:len_zip:down],
    ):

        if i % 5000 == 0:
            print(i / len(start_index[0:len_zip:down]))
        [attention_random[start, :], _] = np.histogram(
            tarraw[tarstart:tarend], num_bin, range=(lower_bin, upper_bin), density=True
        )
        sample = np.random.randint(low=tarstart, high=tarend, size=end - start)
        [attention_sample[start, :], _] = np.histogram(
            tarraw[sample], num_bin, range=(lower_bin, upper_bin), density=True
        )
        attention_random_not_set[start] = 1

    # fill empty location: attention random: due to downsampling: speed
    print("\t fill empty attention_random locations")
    attention_random_not_set = np.nonzero(attention_random_not_set == 0)[0]
    attention_random_not_set = attention_random_not_set[attention_random_not_set > 0]
    for index_not_set in attention_random_not_set:
        attention_random[index_not_set, :] = attention_random[index_not_set - 1, :]
        attention_sample[index_not_set, :] = attention_sample[index_not_set - 1, :]

    # attention metric: relative to attention_random: subtract attention_random
    if att_metric:
        print("construct attention metric")
        print(
            "remove reference level, scale [mean=0,std=1], impose hard double-sided threshold"
        )
        print("\t attention")

        # ref to benchmark
        attention = attention - attention_random

        # scale attention
        attention = attention / np.std(attention[attention > 0.0])

        # attention: randomized sampling
        print("\t attention sample")
        # ref to benchmark
        attention_sample = attention_sample - attention_random
        # scale attention_sample
        attention_sample = attention_sample / np.std(
            attention_sample[attention_sample > 0.0]
        )

        # not used
        print("\t attention random")
        # scale attention random
        attention_random = attention_random / np.std(
            attention_random[attention_random > 0.0]
        )

    else:
        print("construct information metric")
        print("scale [std=1], impose hard single-sided threshold")
        # scale attention
        attention = attention / np.std(attention[attention > 0.0])

        # attention: randomized sampling
        print("\t attention sample")

        # scale attention_sample
        attention_sample = attention_sample / np.std(
            attention_sample[attention_sample > 0.0]
        )

        # randomized attention
        print("\t attention random")
        # scale attention random
        attention_random = attention_random / np.std(
            attention_random[attention_random > 0.0]
        )

        # hard threshold
        if hard_threshold_rel_to_mean:
            hard_threshold = np.mean(attention) + 1.0

    # move to equally spaced grid:
    # trick: len(equal_spike_time) = len(start_index): equal_spike_time filled in
    # make equal time: do not need but gives greater clarity
    equal_spike_time = np.linspace(
        np.min(spike_time), np.max(spike_time), len(start_index)
    )
    # map spike_time[start_index] -> equal_spike_index: have repeating spike_time index in equal_spike_index
    equal_spike_index = np.array(
        (spike_time[start_index] - np.min(spike_time))
        / (equal_spike_time[1] - equal_spike_time[0])
        + 0.5
    ).astype(int)

    #
    # create graphics
    #

    #
    # copy random attention: for transparency above/below diastole/systole
    attention_random_copy = np.copy(attention_random)

    print("create graphics")

    hasAttention = True
    plot_results(
        channel_name,
        attention,
        attention_random_copy,
        hard_threshold,
        start_index,
        equal_spike_time,
        equal_spike_index,
        event_name="AllEvents",
        hasLVP=hasLVP,
        hasRESP=hasRESP,
        hasAttention=hasAttention,
        tar_upp_trans=tar_upp_trans,
        tar_low_trans=tar_low_trans,
        lower_bin=lower_bin,
        upper_bin=upper_bin,
        window=window,
        att_metric=att_metric,
        num_bin=num_bin,
        outdir=outdir,
        comment_df=comment_df
        )

    hasAttentionRandom = True
    plot_results(
        channel_name,
        attention_random,
        attention_random_copy,
        hard_threshold,
        start_index,
        equal_spike_time,
        equal_spike_index,
        event_name="AllEvents",
        hasLVP=hasLVP,
        hasRESP=hasRESP,
        hasAttentionRandom=hasAttentionRandom,
        tar_upp_trans=tar_upp_trans,
        tar_low_trans=tar_low_trans,
        lower_bin=lower_bin,
        upper_bin=upper_bin,
        window=window,
        att_metric=att_metric,
        num_bin=num_bin,
        outdir=outdir,
       comment_df=comment_df
       )

    hasAttentionSample = True
    plot_results(
        channel_name,
        attention_sample,
        attention_random_copy,
        hard_threshold,
        start_index,
        equal_spike_time,
        equal_spike_index,
        event_name="AllEvents",
        hasLVP=hasLVP,
        hasRESP=hasRESP,
        hasAttentionSample=hasAttentionSample,
        tar_upp_trans=tar_upp_trans,
        tar_low_trans=tar_low_trans,
        lower_bin=lower_bin,
        upper_bin=upper_bin,
        window=window,
        att_metric=att_metric,
        num_bin=num_bin,
        outdir=outdir,
       comment_df=comment_df
       )

    #loop through start -> end interventions
    #comment_df = pd.read_csv(comment_file)
    f = open(AM_metadatafile, "a+")

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

            try:
                # update event_name to include the number of the event
                event_name = event_name + str(event_num)
                
                f.write('\n LOOP CHECK \n')
                f.write(str(start) + ' ')
                f.write(str(end) + ' ')
                f.write(event_name)
                f.flush()
    
                # global
                # extract start->end event
                # indices
                sub = np.nonzero(np.logical_and(spike_time>start-before_buffer, spike_time<end+after_buffer))[0]
                # times
                spike_timesub=spike_time[sub]
                # attention
                attentionsub=attention[sub,:]
                # random_copy
                attention_random_copysub=attention_random_copy[sub,:]
                
                # local indices: 0,1, ..., length of sub (num times spanning event)
                start_indexsub = np.arange(len(sub))
    
                # move to equally spaced grid:
                # trick: len(equal_spike_timesub) = len(start_indexsub): equal_spike_timesub filled in
                # make equal time: do not need but gives greater clarity
                equal_spike_timesub = np.linspace(
                    np.min(spike_timesub), np.max(spike_timesub), len(start_indexsub)
                )
                # map spike_time[start_index] -> equal_spike_index: have repeating spike_time index in equal_spike_index
                equal_spike_indexsub = np.array(
                    (spike_timesub[start_indexsub] - np.min(spike_timesub))
                    / (equal_spike_timesub[1] - equal_spike_timesub[0])
                    + 0.5
                ).astype(int)            
    
                
                f.write('\n INDEX CHECK \n')
                f.write(str(sub.shape))
                f.write(str(start_indexsub.shape))
                f.write(str(equal_spike_indexsub.shape))
                f.write(str(equal_spike_timesub.shape))
                f.write(str(attentionsub.shape))
                f.write(str(attention_random_copysub.shape))
                f.flush()
    
                hasAttention = True
                plot_results(
                    channel_name,
                    attentionsub,
                    attention_random_copysub,
                    hard_threshold,
                    start_indexsub,
                    equal_spike_timesub,
                    equal_spike_indexsub,
                    event_name=event_name,
                    hasLVP=hasLVP,
                    hasRESP=hasRESP,
                    hasAttention=hasAttention,
                    tar_upp_trans=tar_upp_trans,
                    tar_low_trans=tar_low_trans,
                    lower_bin=lower_bin,
                    upper_bin=upper_bin,
                    window=window,
                    att_metric=att_metric,
                    num_bin=num_bin,
                    outdir=outdir,
                    comment_df=comment_df
                    )
               
                print('Found Event ', event_name)
                f.write('Found Event ')
                f.write(event_name)
                f.flush()
            except:
                print('\n Did Not Find Event ', event_name)
                f.write('\n Did not Find Event ')
                f.write(event_name)
                f.flush()
    except:
        print('\n comment data frame corrupt ')
        f.write('\n comment data frame corrupt ')
        f.flush()
    
    f.close()

