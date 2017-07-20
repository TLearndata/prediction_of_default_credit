#!/usr/bin/env python        直接在Unix/Linux/Mac上运行
# -*- coding: utf-8 -*-      使用标准UTF-8编码；
__author__ = 'Harry Bai'
#导入相关机器学习包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)

#导入数据
data = pd.read_csv("data_原始.csv")

#find the datatype of all the columns
data.dtypes

#对缺失值做处理
data.replace([-1],np.nan).isnull().sum()

#分析好坏样本比例
print ("Fraud")
print (data['paid.ratio'][data.status == 1].describe())
print ()
print ("Normal")
print (data['paid.ratio'][data.status == 0].describe())
print ()

'''
- 初步判断正负样本比例6:1，由于数据量较小，可以采用对负样本重采样的方法
- 且负样本中还款率均值为30%，说明一旦出现未还的账目，借款人就倾向于不还钱（破窗效应）
- 3/4分为点才达到50%的还款率，进一步说明还款率较低
'''

#简单对异常值做分析
def detection(dataframe, threshold=.95):
    d = dataframe['graduated.years']
    dataframe['isAnomaly'] = d > d.quantile(threshold)  
    return dataframe
dd = detection(data)
dd[dd.isAnomaly>0].status.count()

#打标签
label = data.status
#del data['status']
del data['paid.ratio']
del data['group']

#find the datatype of all the columns
data.replace([-1],np.nan).isnull().sum()


#变量相关性图
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

#数据概况
data.head()

#双变量分析
g = sns.pairplot(data[[u'age',  u'gender', u'salary', u'education', u'graduated.years', u'marriageStatus',
       u'hasChild', u'hasHouse']], palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])

#一些分析图
data.officeScale.value_counts().sort_index().plot()

sns.distplot(data['graduated.years'])###偏态分布
plt.show()

sns.distplot(data['age'])###偏态分布
plt.show()

sns.countplot(data['credit'])
plt.show()