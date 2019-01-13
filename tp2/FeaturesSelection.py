from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from numpy import set_printoptions
from numpy import argsort

data = read_csv("Dataset_DecisionPSS.csv",sep=";")
data = data.drop("StudyID", axis=1) 
data = data.dropna(axis=0)
array = data.values
X=array[:,0:17]
Y=array[:,2]

print("US")
test = SelectKBest(score_func=chi2, k=15)
fit = test.fit(X,Y)
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features[0:5,:])
# 14 e 15

print("RFE")
model = LogisticRegression()
rfe = RFE(model,15)
fit = rfe.fit(X,Y)
print(fit.support_)
print(fit.ranking_)
# 2 e 14

print("PCA")
pca=PCA()
fit=pca.fit(X)
print(fit.explained_variance_ratio_)
print(fit.components_)
# 15 e 17

print("VT")
selector = VarianceThreshold()
selector.fit_transform(X)
print(selector.variances_)
# 14 e 15

corr = data.corr()["PSS_Stress"]
print(corr[argsort(corr,axis=0)[::-1]])
# 15 e 2