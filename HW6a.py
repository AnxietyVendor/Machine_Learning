# 模块调用
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 读入数据
Simudata = pd.read_csv('C:/Users/mi/Desktop/Simudata.csv')

Simudata1 = Simudata[Simudata['class'] == 1]
Simudata2 = Simudata[Simudata['class'] == 2]

X1_train, X1_test, y1_train, y1_test = train_test_split(Simudata1.iloc[:,0:2], Simudata1.iloc[:,2], test_size = 0.2)
X2_train, X2_test, y2_train, y2_test = train_test_split(Simudata2.iloc[:,0:2], Simudata2.iloc[:,2], test_size = 0.2)

X_train = X1_train.append(X2_train)
y_train = y1_train.append(y2_train)


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
        return 1
    
    else:
        return 2


avg1, cov_m1 = cov_avg(X1_train)
avg2, cov_m2 = cov_avg(X2_train)
avg, cov_m = cov_avg(X_train)

w = fisher(X1_train, X2_train)

X_test = X1_test.append(X2_test)
y_test = y1_test.append(y2_test)

pred = []
for i in range(X_test.shape[0]):
    label = judge(X_test.iloc[i,:], w, avg1, avg2)
    pred.append(label)



xx = np.arange(-1.5,4,0.01)
yy = xx*w[1]/w[0]

# 预测结果为第一类的点
xp1 = X_test.iloc[list(np.where(np.array(pred) == 1)[0]),0]
yp1 = X_test.iloc[list(np.where(np.array(pred) == 1)[0]),1]

# 预测结果为第二类的点
xp2 = X_test.iloc[list(np.where(np.array(pred) == 2)[0]),0]
yp2 = X_test.iloc[list(np.where(np.array(pred) == 2)[0]),1]

fig, ax = plt.subplots()
line = ax.plot(xx, yy, color = 'black',linewidth = 2)
plt.scatter(xp1, yp1, color = 'red', s = 3)
plt.scatter(xp2, yp2, color = 'green', s = 3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fisher Discriminant Analysis')
plt.show()

print('Accuracy Score of Fisher Discriminant Analysis: %f' %(accuracy_score(y_test,pred)))
# LDA方法
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, y_train)
pred = []
pred = LDA.predict(X_test)
para = LDA.get_params()


w1 = np.dot(np.linalg.inv(cov_m), (avg1 - avg2).T)
w0 = -0.5*np.dot((avg1 + avg2).T, w1)

xx = np.arange(-10,40,0.01)
yy = -xx*w1[0]/w1[1] - w0

# 预测结果为第一类的点
xp1 = X_test.iloc[list(np.where(np.array(pred) == 1)[0]),0]
yp1 = X_test.iloc[list(np.where(np.array(pred) == 1)[0]),1]

# 预测结果为第二类的点
xp2 = X_test.iloc[list(np.where(np.array(pred) == 2)[0]),0]
yp2 = X_test.iloc[list(np.where(np.array(pred) == 2)[0]),1]

fig, ax = plt.subplots()
line = ax.plot(xx, yy, color = 'black',linewidth = 2)
plt.scatter(xp1, yp1, color = 'red', s = 3)
plt.scatter(xp2, yp2, color = 'green', s = 3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Discriminant Analysis')
plt.show()

print('Accuracy Score of LDA: %f' %(accuracy_score(y_test,pred)))
