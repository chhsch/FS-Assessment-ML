# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:47:01 2021

@author: tingyanglu
"""

#%% import module
import pandas as pd
import numpy as np
from numpy import mean
from os import listdir
from scipy.signal import find_peaks
from smoothness import log_dimensionless_jerk as ldlj
from smoothness import sparc

#%% feature function
def duration (df):
    """
    duration(sec) : len of dataframe(samplepoints)/128(samplepoints/sec)
    """
    dur = len(df)/128;
    
    return dur

def RAV(df, r_gyr_cols):
    """
    range of angular velocity : average of tri axial gyroscope
    """
    r = []
    for col in r_gyr_cols:
        r.append (max(df[col]) - min(df[col]))
    rav = mean(r)
    
    return rav

def P_index(df, p_acc_cols, p_gyr_cols):
    """
    power index : the product of the acceleration range by the angular velocity range
    """
    r_acc = []
    r_gyr = []
    for col in p_acc_cols:
        r_acc.append (max(df[col]) - min(df[col]))
    for col in p_gyr_cols:
        r_gyr.append (max(df[col]) - min(df[col]))
    
    p = sum([a*b for a,b in zip (r_acc, r_gyr)])
    
    return p

def peak_count(df,pc_N_col):
    """
    peak count : use find peak function, get peak index then count the amount of peak 
    """
    pc, _ = find_peaks(df[pc_N_col])
    
    return len(pc)
  
def mean_crossing_count(df,mc_col):
    """
    mean crossing point count : df subtract df.mean then count zero crossing point
    """
    mcc = len(np.where(np.diff(np.sign(df[mc_col]-mean(df[mc_col]))))[0])
    
    return mcc

#%% generate feature by file
filepath = 'D:/UCLAB_D/Frozen shoulder data collection/Checked/5_task_sub'
file = np.array(listdir(filepath))
# file = np.reshape(file,[-1,4])
# file = file[:8]
FT = np.empty([file.shape[0],13])

# 0 for Wrist, 1 for Arm
usecols={0:[0, 1, 2, 6, 7, 8, 9, 13],
         1:[14, 15, 16, 20, 21, 22, 23, 27]}

rav_cols= [4, 5, 6]
pa_cols, pg_cols = [0, 1, 2], rav_cols

for idx, filename in enumerate(file):
    # print(filename)
    # print(type(idx))
    ls_ft = []
    for i in range(2):
        # print(i)
        df = pd.read_csv(f'{filepath}/{filename}', usecols = usecols[i])
        df_col = df.columns
        # print(RAV(df, df_cols[rav_cols]))
        ls_ft.append( RAV(df, df_col[rav_cols]) )
        ls_ft.append( P_index(df, df_col[pa_cols], df_col[pg_cols]) )
        ls_ft.append( ldlj(np.array([df[df_col[3]].to_numpy()]).T, fs = 128., data_type = 'accl', rem_mean = True) )
        ls_ft.append( mean_crossing_count (df,df_col[3]) )
        ls_ft.append( peak_count (df,df_col[3]) )
        ls_ft.append( sparc(df[df_col[7]].to_numpy(), fs=128.)[0] )
    ls_ft.append( duration(df) )
    FT[idx,:]=ls_ft

#%% generate df_FT column name       
columns = ['Wrist_RAV', 'Wrist_PI', 'Wrist_Acc_Norm_LDLJA',
            'Wrist_Acc_Norm_Mcc', 'Wrist_Acc_Norm_Pc', 'Wrist_Gyr_Norm_SPARC',
            'Arm_RAV', 'Arm_PI', 'Arm_Acc_Norm_LDLJA',
            'Arm_Acc_Norm_Mcc', 'Arm_Acc_Norm_Pc', 'Arm_Gyr_Norm_SPARC',
            'Duration']
# columne name with subtask order
new_columns = []
for i in range(4):
    for col in columns:
        if i ==0 :
            new_columns.append(col)
        else:
            new_columns.append(f'{col}_Sub{i}')

#%% generate df_FT & save feature table         
FT = np.reshape(FT, [-1, 52])
# df_FT = pd.DataFrame(FT) 
df_FT = pd.DataFrame(FT, columns = new_columns)
df_FT.insert(0, "Filename", file[0::4]) 
df_FT.to_csv(f'{filepath}/FT.csv', index=False) 