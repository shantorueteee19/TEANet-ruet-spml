import math
import numpy as np
import pandas as pd
def minorityClass_aug(df, window_len, samp_freq, n_majority, label):
  df = df.iloc[:, 0:-1]
  reshaped_df = df.values.reshape(1, df.shape[0]*df.shape[1])
  reshaped_df = pd.DataFrame(reshaped_df)
  diff = ((reshaped_df.shape[1]-window_len*samp_freq)/(n_majority-1)) #formula to calculate the difference between two successive start or end indices
  d = math.floor(diff) #manages floaing point
  start_index = 0
  end_index = window_len*samp_freq
  df_aug = pd.DataFrame()
  for loov in range(n_majority):
    data = reshaped_df.iloc[:, start_index:end_index].values
    data = pd.DataFrame(data)
    df_aug = pd.concat([df_aug, data], axis=0, ignore_index=True)
    start_index = start_index + d
    end_index = end_index + d
  y = pd.DataFrame([label]*df_aug.shape[0])
  df_aug = pd.concat([df_aug, y], axis=1, ignore_index=True)
  return df_aug
