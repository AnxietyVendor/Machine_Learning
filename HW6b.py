import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# OvO
Boston = pd.read_csv('C:/Users/mi/Desktop/Boston.csv')
percentile = np.percentile(Boston['medv'], (25,50,75), interpolation = 'midpoint') 

medv1 = Boston[Boston['medv'] < percentile[0]]
medv2 = Boston[Boston['medv'] >= percentile[0]][Boston['medv'] < percentile[1]]
medv3 = Boston[Boston['medv'] >= percentile[1]][Boston['medv'] < percentile[2]]
medv4 = Boston[Boston['medv'] >= percentile[2]]

medv1['class'] = 1
medv2['class'] = 2
medv3['class'] = 3
medv4['class'] = 4

X1_train, X1_test, y1_train, y1_test = train_test_split(medv1[['industry','river','nox','rooms','age','distance']],medv1['class'] , test_size = 0.2)
X2_train, X2_test, y2_train, y2_test = train_test_split(medv2[['industry','river','nox','rooms','age','distance']],medv2['class'] , test_size = 0.2)
X3_train, X3_test, y3_train, y3_test = train_test_split(medv3[['industry','river','nox','rooms','age','distance']],medv3['class'] , test_size = 0.2)
X4_train, X4_test, y4_train, y4_test = train_test_split(medv4[['industry','river','nox','rooms','age','distance']],medv4['class'] , test_size = 0.2)

LDA1 = LinearDiscriminantAnalysis()
LDA2 = LinearDiscriminantAnalysis()
LDA3 = LinearDiscriminantAnalysis()
LDA4 = LinearDiscriminantAnalysis()
LDA5 = LinearDiscriminantAnalysis()
LDA6 = LinearDiscriminantAnalysis()

X12_train = X1_train.append(X2_train)
y12_train = y1_train.append(y2_train)
LDA1.fit(X12_train, y12_train)

X13_train = X1_train.append(X3_train)
y13_train = y1_train.append(y3_train)
LDA2.fit(X13_train, y13_train)

X14_train = X1_train.append(X4_train)
y14_train = y1_train.append(y4_train)
LDA3.fit(X14_train, y14_train)

X23_train = X2_train.append(X3_train)
y23_train = y2_train.append(y3_train)
LDA4.fit(X23_train, y23_train)

X24_train = X2_train.append(X4_train)
y24_train = y2_train.append(y4_train)
LDA5.fit(X24_train, y24_train)

X34_train = X3_train.append(X4_train)
y34_train = y3_train.append(y4_train)
LDA6.fit(X34_train, y34_train)

X_test = X1_test.append([X2_test,X3_test,X4_test])
y_test = y1_test.append([y2_test,y3_test,y4_test])

pred = []
for i in range(X_test.shape[0]):
    y_pred = [LDA1.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDA2.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDA3.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDA4.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDA5.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDA6.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0]]
    y_preds = pd.Series(data = y_pred)
    pred.append(y_preds.mode()[0])
    
print('Accuracy Score of LDA OvO: %f' %(accuracy_score(y_test,pred)))

# MvM
LDAECOC1 = LinearDiscriminantAnalysis()
LDAECOC2 = LinearDiscriminantAnalysis()
LDAECOC3 = LinearDiscriminantAnalysis()
LDAECOC4 = LinearDiscriminantAnalysis()
LDAECOC5 = LinearDiscriminantAnalysis()

f_train = X1_train.append([X2_train,X3_train,X4_train])

# f1训练数据
ECOC11_train= [-1]*X1_train.shape[0]
ECOC12_train =  [1]*X2_train.shape[0]
ECOC13_train= [-1]*X3_train.shape[0]
ECOC14_train = [-1]*X4_train.shape[0]

f1_ECOC = ECOC11_train + ECOC12_train + ECOC13_train + ECOC14_train
LDAECOC1.fit(f_train, f1_ECOC)

# f2训练数据
ECOC21_train= [1]*X1_train.shape[0]
ECOC22_train = [-1]*X2_train.shape[0]
ECOC23_train= [1]*X3_train.shape[0]
ECOC24_train = [-1]*X4_train.shape[0]

f2_ECOC = ECOC21_train + ECOC22_train + ECOC23_train + ECOC24_train
LDAECOC2.fit(f_train, f2_ECOC)

# f3训练数据
ECOC31_train= [-1]*X1_train.shape[0]
ECOC32_train =[-1]*X2_train.shape[0]
ECOC33_train= [1]*X3_train.shape[0]
ECOC34_train =[1]*X4_train.shape[0]

f3_ECOC = ECOC31_train + ECOC32_train + ECOC33_train + ECOC34_train
LDAECOC3.fit(f_train, f3_ECOC)

# f4训练数据
ECOC41_train= [1]*X1_train.shape[0]
ECOC42_train =[1]*X2_train.shape[0]
ECOC43_train= [-1]*X3_train.shape[0]
ECOC44_train =[1]*X4_train.shape[0]

f4_ECOC = ECOC41_train + ECOC42_train + ECOC43_train + ECOC44_train
LDAECOC4.fit(f_train, f4_ECOC)

# f5训练数据
ECOC51_train= [1]*X1_train.shape[0]
ECOC52_train =[-1]*X2_train.shape[0]
ECOC53_train= [1]*X3_train.shape[0]
ECOC54_train =[-1]*X4_train.shape[0]

f5_ECOC = ECOC51_train + ECOC52_train + ECOC53_train + ECOC54_train
LDAECOC5.fit(f_train, f5_ECOC)

vector1 = np.array([-1, 1, -1, 1, 1])
vector2 = np.array([1, -1, -1, 1, -1])
vector3 = np.array([-1, 1, 1, -1, 1])
vector4 = np.array([-1, -1, 1, -1, 1])

pred = []
for i in range(X_test.shape[0]):
    y_pred = np.array([LDAECOC1.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDAECOC2.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDAECOC3.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDAECOC4.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0],LDAECOC5.predict(np.array(X_test.iloc[i,:]).reshape(1, -1))[0]])
    hm_dist1 = np.linalg.norm(vector1 - y_pred, ord = 1)
    eu_dist1 = np.linalg.norm(vector1 - y_pred)
    hm_dist2 = np.linalg.norm(vector2 - y_pred, ord = 1)
    eu_dist2 = np.linalg.norm(vector2 - y_pred)
    hm_dist3 = np.linalg.norm(vector3 - y_pred, ord = 1)
    eu_dist3 = np.linalg.norm(vector3 - y_pred)
    hm_dist4 = np.linalg.norm(vector4 - y_pred, ord = 1)
    eu_dist4 = np.linalg.norm(vector4 - y_pred)
    
    hm_dist = [hm_dist1,hm_dist2,hm_dist3,hm_dist4]
    eu_dist = [eu_dist1,eu_dist2,eu_dist3,eu_dist4]
    
    pred.append(eu_dist.index(min(eu_dist)) + 1)

print('Accuracy Score of LDA MvM: %f' %(accuracy_score(y_test,pred)))
