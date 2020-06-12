from numpy import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# 划分数据集
#X = melon_dataset.iloc[:,1:-1]
# 西瓜数据集3.0a(忽略离散属性)
X = melon_dataset.loc[:,['密度', '含糖率']]
y = melon_dataset.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4,stratify = y, random_state = 1)

clf = LDA()
clf.fit(X,y)

x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1

xx, yy = meshgrid(arange(x_min, x_max, 0.01), arange(y_min, y_max, 0.01))
xxx = arange(x_min, x_max, 0.01)
yyy = (-clf.coef_[0][0] * xxx - clf.intercept_)/ clf.coef_[0][1]

Z = clf.predict(c_[xx.ravel(), yy.ravel()])
Z = (Z.reshape(xx.shape) == '否').astype(int)

plt.contourf(xx,yy,Z, cmap = plt.cm.Spectral)
plt.plot(xxx,yyy,color = 'black')
plt.plot()
plt.scatter(X.iloc[:,0], X.iloc[:,1], cmap = plt.cm.Spectral)
plt.axis('tight')
plt.show()

print(clf.intercept_)
print(clf.coef_)