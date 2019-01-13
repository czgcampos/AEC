import sklearn.datasets
wine = sklearn.datasets.load_wine()
from sklearn.svm import SVC
svm=SVC()
svm.fit(wine.data,wine.target);
svm.score(wine.data,wine.target)