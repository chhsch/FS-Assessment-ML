#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:41:43 2022

@author: hathaway
"""

import pandas as pd
import numpy as np
import pingouin as pg
df = pd.read_excel('/Users/hathaway/Desktop/mild vs moderate severe.xlsx')

head=["POH_Wrist_RAV","POH_Wrist_PI","POH_Wrist_Acc_Norm_LDLJA","POH_Wrist_Acc_Norm_Mcc","POH_Wrist_Acc_Norm_Pc","POH_Wrist_Gyr_Norm_SPARC","POH_Arm_RAV","POH_Arm_PI","POH_Arm_Acc_Norm_LDLJA","POH_Arm_Acc_Norm_Mcc","POH_Arm_Acc_Norm_Pc","POH_Arm_Gyr_Norm_SPARC","POH_Duration","ROP_Wrist_RAV","ROP_Wrist_PI","ROP_Wrist_Acc_Norm_LDLJA","ROP_Wrist_Acc_Norm_Mcc","ROP_Wrist_Acc_Norm_Pc","ROP_Wrist_Gyr_Norm_SPARC","ROP_Arm_RAV","ROP_Arm_PI","ROP_Arm_Acc_Norm_LDLJA","ROP_Arm_Acc_Norm_Mcc","ROP_Arm_Acc_Norm_Pc","ROP_Arm_Gyr_Norm_SPARC","ROP_Duration","WH_Wrist_RAV","WH_Wrist_PI","WH_Wrist_Acc_Norm_LDLJA","WH_Wrist_Acc_Norm_Mcc","WH_Wrist_Acc_Norm_Pc","WH_Wrist_Gyr_Norm_SPARC","WH_Arm_RAV","WH_Arm_PI","WH_Arm_Acc_Norm_LDLJA","WH_Arm_Acc_Norm_Mcc","WH_Arm_Acc_Norm_Pc","WH_Arm_Gyr_Norm_SPARC","WH_Duration","WLB_Wrist_RAV","WLB_Wrist_PI","WLB_Wrist_Acc_Norm_LDLJA","WLB_Wrist_Acc_Norm_Mcc","WLB_Wrist_Acc_Norm_Pc","WLB_Wrist_Gyr_Norm_SPARC","WLB_Arm_RAV","WLB_Arm_PI","WLB_Arm_Acc_Norm_LDLJA","WLB_Arm_Acc_Norm_Mcc","WLB_Arm_Acc_Norm_Pc","WLB_Arm_Gyr_Norm_SPARC","WLB_Duration","WUB_Wrist_RAV","WUB","WUB_Wrist_Acc_Norm_LDLJA","WUB_Wrist_Acc_Norm_Mcc","WUB_Wrist_Acc_Norm_Pc","WUB_Wrist_Gyr_Norm_SPARC","WUB_Arm_RAV","WUB_Arm_PI","WUB_Arm_Acc_Norm_LDLJA","WUB_Arm_Acc_Norm_Mcc","WUB_Arm_Acc_Norm_Pc","WUB_Arm_Gyr_Norm_SPARC","WUB_Duration"]
def pgenerate(df,i):
    aov = pg.anova(dv=head[i], between='Qn_group', data=df, detailed=True)
    ano=aov["p-unc"][0]
    return ano




def ptable(ano,p_table_zero):
    table=np.vstack((p_table_zero,ano))
    return table
    
def main():
    p_table_zero= np.zeros((1,)) 
    for i in range(0,65):
        ano=pgenerate(df,i)
        table=ptable(ano,p_table_zero)
        p_table_zero=table
    return table
table=main()
A=np.delete(table,0, axis = 0)#把table第一列刪除
headay=np.array(head)
B=np.append(headay,A)#為了把feature名字與結果合併
C=pd.DataFrame(B)
filepath='/Users/hathaway/Desktop/'

new=C.to_csv(f'{filepath}/new.csv',index=True)      