from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing

print(__doc__)

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

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.8)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=2,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    
    print(clf.best_params_)
    parametros = clf.best_params_
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    new=clf.predict(X_test)
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
