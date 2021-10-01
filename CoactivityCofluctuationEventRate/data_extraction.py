import os
import numpy as np
import pandas as pd

for channel in sorted(os.listdir(os.getcwd())[:-1]):
    channel_path = os.getcwd() + '\\' + channel + '\\'
    filename = 'outputSpike_uniform_time_rate_Animal_' + channel.split('_')[-1] + '.csv'
    df = pd.read_csv(channel_path + filename)
    new_df = df[np.logical_and(df['time'] >= 11000, df['time'] <= 11120)]
    cols = ['time', ' rate']
    new_df = new_df[cols]
    new_df.to_csv(channel_path + filename)