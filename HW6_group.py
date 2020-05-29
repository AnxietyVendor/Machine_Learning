# 模块调用
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
# 读入数据
Smarket_data = pd.read_csv('C:/Users/mi/Desktop/Smarket.csv')

# 绘制各个变量的散点图
#pd.plotting.scatter_matrix(Smarket_data, figsize = (30,30))

le = LabelEncoder()

# 选取direction作为因变量
y = le.fit_transform(Smarket_data['Direction'].tolist())

############################################################################### EXPERIMENT 1: LDA,QDA lag1 lag2 -> direction
# 选取Lag1和Lag2作为特征
# 选择2001-2004的数据作为训练数据，2005的数据作为测试数据

X = np.mat([Smarket_data['Lag1'].tolist(), Smarket_data['Lag2'].tolist()]).T
time = Smarket_data['Year'].tolist()

len_train = sum(np.array(time) < 2005)

# 划分测试集与训练集
y_train, y_test, X_train, X_test = y[:len_train], y[len_train:], X[:len_train, :], X[len_train :,:]


# LDA模型
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, y_train)
y_pred = LDA.predict(X_test)
prob_pred_LDA = LDA.predict_proba(X_test)
LDA_confusion_matrix = confusion_matrix(y_test, y_pred, labels = [0,1])

print('LDA confusion matrix')
print(LDA_confusion_matrix)
print(' Accuracy Score: %f\n Precision Score: %f \n Recall Score: %f\n F1 Score: %f \n'%(accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))

## 绘制该LDA模型的ROC曲线
plt.figure(1)
plt.title('ROC Curve of LDA Model')
plt.xlabel('FPR')
plt.ylabel('TPR')

fpr,tpr,threshold = roc_curve(y_test, prob_pred_LDA[:,1])
plt.plot(tpr, fpr)
plt.show()

## 输出AUC的值
roc_auc_1 = auc(fpr,tpr) 
print('AUC of Model 1 is ' + roc_auc_1.astype(str))



# 在LDA的基础上改变阈值，观察性能指标的变化
threshold = 0.48
prob_pred = LDA.predict_proba(X_test)
y_pred = (prob_pred[:, 1] > threshold).astype(int)
LDA_confusion_matrix = confusion_matrix(y_test, y_pred, labels = [0,1])

print('LDA confusion matrix')
print(LDA_confusion_matrix)
print(' Accuracy Score: %f\n Precision Score: %f \n Recall Score: %f\n F1 Score: %f \n'%(accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))



# QDA模型
QDA = QuadraticDiscriminantAnalysis()
QDA.fit(X_train, y_train)
y_pred = QDA.predict(X_test)
prob_pred_QDA = QDA.predict_proba(X_test)
QDA_confusion_matrix = confusion_matrix(y_test, y_pred, labels = [0,1])

print('QDA confusion matrix')
print(QDA_confusion_matrix)
print(' Accuracy Score: %f\n Precision Score: %f \n Recall Score: %f\n F1 Score: %f \n'%(accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))

## 绘制该QDA模型的ROC曲线
plt.figure(1)
plt.title('ROC Curve of QDA Model')
plt.xlabel('FPR')
plt.ylabel('TPR')

fpr,tpr,threshold = roc_curve(y_test, prob_pred_QDA[:,1])
plt.plot(tpr, fpr)
plt.show()

## 输出AUC的值
roc_auc_1 = auc(fpr,tpr) 
print('AUC of Model 1 is ' + roc_auc_1.astype(str))


# Logistic回归模型

clf=LogisticRegression(random_state=0).fit(X_train,y_train)
print(clf.intercept_,clf.coef_)

y_pred=clf.predict(X_test)

C2=confusion_matrix(y_test,y_pred,labels=[0,1])

print('Logistic confusion matrix')
print(C2)

print(' Accuracy Score: %f\n Precision Score: %f \n Recall Score: %f\n F1 Score: %f \n'\
      %(accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), \
        f1_score(y_test, y_pred)))
    



plt.figure(1)
plt.title('ROC Curve of Logistic Model')
plt.xlabel('FPR')
plt.ylabel('TPR')

fpr,tpr,threshold = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.plot(tpr, fpr)
plt.show()
roc_auc_1 = auc(fpr,tpr) 
print('AUC of Model 1 is ' + roc_auc_1.astype(str))


# Fisher判别分析
Smarket_data0 = Smarket_data[Smarket_data['Direction'] == 'Down' ]
Smarket_data1 = Smarket_data[Smarket_data['Direction'] == 'Up' ]


le = LabelEncoder()

# 选取direction作为因变量
y0 = le.fit_transform(Smarket_data0['Direction'].tolist())
y1 = le.fit_transform(Smarket_data1['Direction'].tolist())
############################################################################### EXPERIMENT 1: LDA,QDA lag1 lag2 -> direction
# 选取Lag1和Lag2作为特征
# 选择2001-2004的数据作为训练数据，2005的数据作为测试数据

X0 = np.mat([Smarket_data0['Lag1'].tolist(), Smarket_data0['Lag2'].tolist()]).T
X1 = np.mat([Smarket_data1['Lag1'].tolist(), Smarket_data1['Lag2'].tolist()]).T

time = Smarket_data['Year'].tolist()

len_train = sum(np.array(time) < 2005)

# 划分测试集与训练集
y0_train, y0_test, X0_train, X0_test = y0[:len_train], y0[len_train:], X0[:len_train, :], X0[len_train :,:]
y1_train, y1_test, X1_train, X1_test = y1[:len_train], y1[len_train:], X1[:len_train, :], X1[len_train :,:]


#
Train_data = Smarket_data[Smarket_data['Year'] < 2005 ]
Test_data = Smarket_data[Smarket_data['Year'] == 2005 ]

Train_data0 = Train_data[Train_data['Direction'] == 'Down']
Train_data1 = Train_data[Train_data['Direction'] == 'Up']

Test_data0 = Test_data[Test_data['Direction'] == 'Down']
Test_data1 = Test_data[Test_data['Direction'] == 'Up']


y0_train = (Train_data0['Direction'] == 'Up').astype(int)
y1_train = (Train_data1['Direction'] == 'Up').astype(int)

X0_train = Train_data0[['Lag1','Lag2']]
X1_train = Train_data1[['Lag1','Lag2']]

y0_test = (Test_data0['Direction'] == 'Up').astype(int)
y1_test = (Test_data1['Direction'] == 'Up').astype(int)

X0_test = Test_data0[['Lag1','Lag2']]
X1_test = Test_data1[['Lag1','Lag2']]

X_train = X0_train.append(X1_train)
y_train = y0_train.append(y1_train)

X_test = X0_test.append(X1_test)
y_test = y0_test.append(y1_test)

# 函数设计
def cov_avg(X):
    # 计算给定数据的均值和协方差阵
    
    row, col = X.shape
    x_bar = np.mean(X)
    cov_m = np.dot((X - np.mean(X)).T, (X - np.mean(X)))/row
    return x_bar, cov_m
    
def fisher(X1, X2):
    # Fisher算法
    avg1, cov_m1 = cov_avg(X1)
    avg2, cov_m2 = cov_avg(X2)
    Sw = cov_m1 + cov_m2
    u, s, v = np.linalg.svd(Sw)
    Sw_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return np.dot(Sw_inv, avg1 - avg2)
    
def judge(X, w, avg1, avg2):

    pos = np.dot(X, w)
    center1 = np.dot(avg1, w)
    center2 = np.dot(avg2, w)
    
    dist1 = np.linalg.norm(pos - center1)
    dist2 = np.linalg.norm(pos - center2)
    
    if dist1 < dist2:
        return 0
    
    else:
        return 1


avg1, cov_m1 = cov_avg(X0_train)
avg2, cov_m2 = cov_avg(X1_train)
avg, cov_m = cov_avg(X_train)

w = fisher(X0_train, X1_train)


y_pred = []
for i in range(X_test.shape[0]):
    label = judge(X_test.iloc[i,:], w, avg1, avg2)
    y_pred.append(label)

Fisher_confusion_matrix = confusion_matrix(y_test, y_pred, labels = [0,1])
print('Fisher confusion matrix')
print(Fisher_confusion_matrix)
print(' Accuracy Score: %f\n Precision Score: %f \n Recall Score: %f\n F1 Score: %f \n'%(accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))
