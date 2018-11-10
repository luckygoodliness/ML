import  numpy  as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn import datasets
# 导入鸢尾花数据集
iris = datasets.load_iris()

'''
大写 X
'''
X = iris.data
y = iris.target

from playML.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,seed=666)



def plot_decision_boundary(model,axis):
    x0,x1 = np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)),
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100))
    )
    X_new = np.c_[x0.ravel(),x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    plt.contourf(x0,x1,zz,linewidth=5,cmap=custom_cmap)

'''
1.iris数据集中有三类数据，所以要全部类别提取，三类出来进行训练
2.为了平面画图说明，只取出前两个特征。取出两个特征可以作图展示
'''
'''
knn中没有决策边界表达式，但是仍然可以通过这种方式进行绘制
'''
from sklearn.neighbors import KNeighborsClassifier

knn_clf_all = KNeighborsClassifier()
knn_clf_all.fit(iris.data[:,:2],iris.target)

plot_decision_boundary(knn_clf_all,axis=[4,8,1.5,4.5])
plt.scatter(iris.data[iris.target==0,0],iris.data[iris.target==0,1])
plt.scatter(iris.data[iris.target==1,0],iris.data[iris.target==1,1])
plt.scatter(iris.data[iris.target==2,0],iris.data[iris.target==2,1])
plt.show()

