from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr.predict(x_test)
lr.score(x_test,y_test)