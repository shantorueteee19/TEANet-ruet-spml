import scipy.signal as signal
import numpy as np
import pandas as pd
def segment_func(df, seg_length, fs, fs_new): #fs original sampling frequency, fs_new to downsample
    df = df.values
    num_samples_initial = len(df)
    num_samples_down = int(num_samples_initial*fs_new/fs)
    df_down = signal.resample(df, num_samples_down)
    rows = int(num_samples_down//(fs_new*seg_length))
    cols = fs_new*seg_length
    df_reshaped = df_down[:cols*rows].reshape(rows, cols)
    df_reshaped = pd.DataFrame(df_reshaped, columns=None)
    return df_reshaped

def label_func(df, label):
    rows = df.shape[0]
    arr = pd.DataFrame(np.zeros((rows, 1)), columns=None)
    arr.iloc[:, 0] = label
    df = pd.concat([df,arr], axis = 1, ignore_index = True )
    return df