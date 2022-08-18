#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:32:44 2022

@author: hathaway
"""

import numpy as np
import pandas as pd



df = pd.read_excel('/Users/hathaway/Desktop/normalize(舊data)第一次.xlsx')
#df.loc[(df.Qn<16),["Qn_group"]]=1
#df.loc[(df.Qn>16),["Qn_group"]]=2
#df.loc[(df.Qn<34),["Qn_group"]]=1
#df.loc[(df.Qn>34),["Qn_group"]]=2
#df.loc[(df.Qn==34),["Qn_group"]]=1
df.loc[(df.Qn<16),["Qn_group"]]=1
df.loc[(df.Qn==16),["Qn_group"]]=1
#df.loc[(df.Qn>17),["Qn_group"]]=2
#df.loc[(df.Qn==17),["Qn_group"]]=2
df.loc[(df.Qn>34),["Qn_group"]]=2
df.loc[(df.Qn==34),["Qn_group"]]=2
df.to_excel("/Users/hathaway/Desktop/severe vs mild moderate.xlsx" ,index =False)

#model = ols("POH_Wrist_RAV",data =df).fit()
B_corr= df.corr(method = 'spearman')
A=B_corr.loc["Qn_group"]
filepath='/Users/hathaway/Desktop/'
new=A.to_csv(f'{filepath}/new.csv',index=True)
#anova_result = sm.stats.anova_lm(model,typ=2)
#print (anova_result)

#excel_file = Workbook()
#df.loc[(df.Qn==16),["Qn_group"]]=1
#class1 = df[df['Qn_group'] == 1]
#class2 = df[df['Qn_group'] == 2]

#df.loc[(df.Qn<10),["Qn_group"]]=1
#df.loc[(df.Qn>10),["Qn_group"]]=2
#df.loc[(df.Qn==10),["Qn_group"]]=1

#df.loc[(df.Qn<16),["Qn_group"]]=2
#df.loc[(df.Qn==16),["Qn_group"]]=2
#df.loc[(df.Qn>17),["Qn_group"]]=1
#df.loc[(df.Qn==17),["Qn_group"]]=1
#df.loc[(df.Qn>34),["Qn_group"]]=2
#df.loc[(df.Qn==34),["Qn_group"]]=2
B_corr= df.corr(method = 'spearman')
filepath='/Users/hathaway/Desktop/'
new=B_corr.to_csv(f'{filepath}/new.csv',index=True)