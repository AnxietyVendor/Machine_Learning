import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
# 读入数据
glass_data = pd.read_excel('C:/Users/mi/Desktop/glass.xlsx')
X_raw = glass_data.iloc[:,1:10]
# 对X做标准化处理
X_scaled = preprocessing.scale(X_raw)
y = glass_data['Type']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.25)
# 训练神经网络
neurons = MLPClassifier(hidden_layer_sizes=(1,2),solver='sgd')
neurons.fit(X_train,y_train)
#print(neurons.score(X_test,y_test))

# 训练逻辑回归模型
logi = LogisticRegression()
logi.fit(X_train,y_train)
#print(logi.score(X_test,y_test))

def neuro_acc(times,nodes):
    '''
    :params times: 模型重复训练次数
    :return accArray: 多次测试得到的得分
    '''
    accArr = []
    for i in range(times):
        #neuron = MLPClassifier(random_state=0,hidden_layer_sizes=(1,nodes),solver='sgd')
        neuron = MLPClassifier(hidden_layer_sizes=(1,nodes),activation = 'sigmoid',solver='sgd')
        neuron.fit(X_train,y_train)
        accArr.append(neuron.score(X_test,y_test))
    return accArr

def logi_acc(times):
    '''
    :params times: 模型重复训练次数
    :return accArray: 多次测试得到的得分
    '''
    accArr = []
    for i in range(times):
        logi = LogisticRegression()
        logi.fit(X_train,y_train)
        accArr.append(logi.score(X_test,y_test))
    return accArr
        
accArr2 = neuro_acc(30,2)
accArr3 = neuro_acc(30,3)
accArr4 = neuro_acc(30,4)
accArr5 = neuro_acc(30,5)

accArrlogi = logi_acc(30)
times = [x+1 for x in range(30)]


fig, ax = plt.subplots()
ax.plot(times,accArr2,label='2 nodes')
ax.plot(times,accArr3,label='3 nodes')
ax.plot(times,accArr4,label='4 nodes')
ax.plot(times,accArr5,label='5 nodes')
ax.plot(times,accArrlogi,color='black',label='Logit Regression')
ax.set_xlabel('Model Number')
ax.set_ylabel('Accuracy')
ax.set_title('Test Accuracy of 1-layer Neuro Network and Logit Regression')
ax.legend()