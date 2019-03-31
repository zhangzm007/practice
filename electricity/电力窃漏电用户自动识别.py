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

#BP神经网络训练模型
from keras.models import Sequential
from keras.layers.core import Dense,Activation
model = Sequential()
model.add(Dense(input_dim = 3,output_dim = 10))
model.add(Activation('relu'))
model.add(Dense(input_dim = 10,output_dim = 1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = ['accuracy'])#指定标准，默认为损失
model.fit(train[:,:3],train[:,3],batch_size = 10,epochs = 100)
m_loss,m_accuracy = model.evaluate(test[:,:3],test[:,3])#返回为损失值和编译中指定的标准
model.save_weights(u'D:\\GitHub\\practice\\electricity\\m.model')

from cm_plot import *
cm_plot(test[:,3],model.predict_classes(test[:,:3]).reshape(len(test))).show() 
#keras里predict_classes得到预测分类结果，predict得到为正结果的概率

#CART决策树
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(train[:,:3],train[:,3])
tree.predict(test[:,:3])
cm_plot(test[:,3],tree.predict(test[:,:3]).reshape(len(test))).show()

from sklearn.externals import joblib
joblib.dump(tree,r'D:\GitHub\practice\electricity\t.pkl')
#sklearn里predict得到预测结果，predict_proba得到结果为正的概率

#模型评价（ROC曲线）
from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(test[:,3],model.predict(test[:,:3]).reshape(len(test)),pos_label = 1)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)

t_fpr,t_tpr,t_thresholds = roc_curve(test[:,3],tree.predict_proba(test[:,:3])[:,1],pos_label = 1)
plt.plot(t_fpr,t_tpr)
plt.ylim(0,1.05)
plt.show()

