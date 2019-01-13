import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing,linear_model
from sklearn.model_selection import train_test_split
  
approxLR = []
lr = linear_model.LinearRegression()

i=0
while i < 100: #numero de vezes que o modelo é medido.
    i+= 1
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
    data_scaler=preprocessing.MinMaxScaler(feature_range=(0,1)) 
    X=data_scaler.fit_transform(X) 

    #Create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    lr.fit(X_train, y_train)    
    predictLR = pd.DataFrame(lr.predict(X_test), columns=['PSS_Stress'])    
    roundLR = np.around(predictLR)
    approxLR.append(accuracy_score(y_test, roundLR))
    
mediaLR = np.mean(approxLR)
print(mediaLR)