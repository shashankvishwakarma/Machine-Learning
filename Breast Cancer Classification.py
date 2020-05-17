import numpy as np
import pandas as pd

# import dataset fro sklearn datasets
from sklearn.datasets import load_breast_cancer

# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

# Making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)

# instance a new object of data
cancer = load_breast_cancer()
print(cancer.keys())
#print(cancer['DESCR'])
#print(cancer['target'])
#print(cancer['target_names'])
#print(cancer['feature_names'])

print(cancer['data'].shape)

# Convert dataset to Dataframe using Panda
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']) )
print(df_cancer.head())

# MODEL TRAINING (FINDING A SOLUTION)

# drop the target column from the dataset to create the input of the model
X = df_cancer.drop(['target'], axis = 1)
# create the expected output
y = df_cancer['target']

# create the expected output
y = df_cancer['target']

# import model for training the model
from sklearn.model_selection import train_test_split

# split data in training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svn_model= SVC()

# Train the model
svn_model.fit(X_train, y_train)

# EVALUATING THE MODEL
# predict the output of our model according using the testing dataset
prediction = svn_model.predict(X_test)
print('Prediction {}'.format(prediction))

# Accuracy
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy: {}".format(accuracy))

# calculate the confusion matrix in order to evaluate the results
cm = confusion_matrix(y_test, prediction)
print('Confusion matrix {}'.format(cm))

# Classification results
report = classification_report(y_test, prediction)
print('Classification results {}'.format(report))
