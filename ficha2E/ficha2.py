import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

columns = "age sex nmi map tc ldl hdl tch ltg glu".split()
diabetes  = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data,columns=columns)
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(df,y,test_size=0.2)
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print ("Score: ", model.score(X_test, y_test))

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

kf = KFold(n_splits=10)
kf.get_n_splits(X_train)
print(kf)

scores = cross_val_score(model, df, y, cv=kf)
print("Cross-validated scores: ",scores)
print("Baseline Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y,predictions)