from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris=datasets.load_iris()
type(iris)
print(iris.keys())
type(iris.data),type(iris.target)
iris.data.shape
iris.target_names
x=iris.data
y=iris.target
df=pd.DataFrame(x,columns=iris.feature_names)
print(df.head())
_ = pd.plotting.scatter_matrix(df, c=y, figsize=[8,8],s=150,marker='D')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'],iris['target'])
iris['data'].shape
iris['target'].shape
prediction = knn.predict(x_new)
X_new.shape
print('Prediction {}'.format(prediction))