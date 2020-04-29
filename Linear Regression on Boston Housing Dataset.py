import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


boston_dataset = load_boston()
print(boston_dataset.keys())
'''
prints -- dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
'''
#print(boston_dataset.feature_names)

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
'''
# Above statement can be split ino 2 as follows
boston = pd.DataFrame(boston_dataset.data)
boston.columns = boston_dataset.feature_names
'''

boston["PRICE"] = boston_dataset.target
#print(boston.describe());

boston.isnull().sum()

Y = boston['PRICE']
X = boston.drop('PRICE', axis=1)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

linearRegression_model = LinearRegression()
linearRegression_model.fit(X_train, Y_train)

#Making prediction for traing data, since we already have the result then could compare the validate the same
y_train_predict = linearRegression_model.predict(X_train)

#Comparing actual value with predicted value, Difference shows error n predication
df = pd.DataFrame(y_train_predict, Y_train)
print(df.head())

from sklearn.metrics import mean_squared_error
#rmse = mean_squared_error(Y_train, y_train_predict)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))

# model evaluation for testing set
y_test_predict = linearRegression_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
