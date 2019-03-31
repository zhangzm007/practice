# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:42:40 2019

@author: Administrator
"""

'''第六章
电力窃漏电用户自动识别'''


import numpy as np
import pandas as pd
data = pd.read_excel(r'C:\Users\Administrator\Desktop\《Python数据分析与挖掘实战》源数据和代码\Python数据分析与挖掘实战\chapter6\demo\data\missing_data.xls',header = None)#header控制字段名
data.columns = ['A','B','C']
#拉格朗日差值填充数据
from scipy.interpolate import lagrange
for i in range(3):
    for j in range(len(data)):
        if pd.isnull(data.iloc[j,i]):
            s = data.iloc[:,i][list(range(j-5,j+6))]#series的序号出现负数
            s = s[s.notnull()]
            data.iloc[j,i] = lagrange(s.index,list(s))(j) #拉格朗日函数返回的是系数,x不需要是连续的,xy必须是一维向量


#构建评价指标后得到专家数据库（已知数据）
model = pd.read_excel(r'C:\Users\Administrator\Desktop\《Python数据分析与挖掘实战》源数据和代码\Python数据分析与挖掘实战\chapter6\demo\data\model.xls')
#随机选择20％数据作为测试数据，其余数据作为训练数据
from random import shuffle
model = model.as_matrix()
shuffle(model)
train = model[:int(0.8*len(model)),:]#训练数据
test = model[int(0.8*len(model)):,:]#测试数据
m_data = model

#LM神经网络训练模型
from keras.models import Sequential
from keras.layers.core import Dense,Activation
model = Sequential()
model.add(Dense(input_dim = 3,output_dim = 10))
model.add(Activation('relu'))
model.add(Dense(input_dim = 10,output_dim = 1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = ['accuracy'])#指定标准，默认为损失
model.fit(train[:,:3],train[:,3],batch_size = 10,epochs = 100)
loss,accuracy = model.evaluate(test[:,:3],test[:,3])#返回为损失值和编译中指定的标准
model.save_weights(u'D:\\Github\\python_practice\\electricity\\m.model')

from cm_plot import *
cm_plot(test[:,3],model.predict_classes(test[:,:3]).reshape(len(test))).show()

