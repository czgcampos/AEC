import pandas as pd
import sklearn as sk
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

data = pd.read_csv("Dataset_DecisionPSS.csv",sep=";")

#filter NaN rows
data = data.dropna(axis=0)

#Feature Selection
data = data.drop("StudyID", axis=1)
data = data.drop("ratio_decisions", axis=1)
data = data.drop("ratio_good_decisions", axis=1)

#Discritization
bins = [0,10,20,30,40,55]
#1-very low 2-low 3-medium 4-high 5-very high
group_names = [1,2,3,4,5]
data['PSS_Stress'] = pd.cut(data['PSS_Stress'],bins,labels=group_names)

y = data.PSS_Stress 
#Class column 
X = data.drop('PSS_Stress', axis=1)

#Normalization
data_scaler=sk.preprocessing.MinMaxScaler(feature_range=(0,1)) 
X=data_scaler.fit_transform(X) 

svm = SVC()
scores = cross_val_score(svm,data,y,cv=10)
print(scores.mean()*100)