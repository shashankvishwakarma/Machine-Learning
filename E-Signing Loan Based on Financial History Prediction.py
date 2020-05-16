# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

# Making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)

#Load Data
dataset = pd.read_csv('./datasets/financial_data.csv')
#print(dataset.head())

# determine the number of records in the dataset
print('The dataset contains {0} rows and {1} columns.'.format(dataset.shape[0], dataset.shape[1]))
#print(df.shape)
#print(df.columns)
#print(df.describe())

# cleaning the data

# Find null columns. If any columns has null values remove that record from dataset
print(dataset.isna().any())
# count how many null values we have in each column
print(dataset.isnull().sum())

# Merge Personal Account Month and Year in one single Personal Account
dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))
#print(dataset[['personal_account_m','personal_account_y','personal_account_months']].head())

# Drop some columns we dont need now
dataset = dataset.drop(columns = ['entry_id','months_employed','personal_account_m','personal_account_y'])
print(dataset.columns)
print(dataset.pay_schedule.value_counts())

# Convert text variables into numerical variable using the Get dummies function
dataset = pd.get_dummies(dataset)
print(dataset.columns)

# Drop some columns we dont need now
dataset = dataset.drop(columns = ['pay_schedule_semi-monthly'])

Y = dataset['e_signed']
X = dataset.drop('e_signed', axis=1)

from sklearn.preprocessing import StandardScaler
# Feature Scaling (Normalize between 0/1). Generate a standard Scalar
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Splitting into train and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size = 0.2, random_state=0)

#Counting and watching the Y_train distribution
print(Y_train.value_counts())

### Model building ###

modelsList = [('Logistic Regression (Lasso)', LogisticRegression(random_state=0, penalty = 'l1', solver='saga')),
              ('Support Vector Machine (Linear)', SVC(random_state=0, kernel = 'linear')),
              ('Support Vector Machine (rbf)', SVC(random_state=0, kernel = 'rbf')),
              ('Random Forest (n=100)',RandomForestClassifier(random_state=0, n_estimators = 100,criterion = 'entropy'))]


# Fitting Model to the Training Set
for i, modelclassifier in modelsList:
    # classifier = LogisticRegression(random_state=0, penalty = 'l1', solver='lbfgs')
    classifier = modelclassifier
    classifier.fit(X_train, Y_train)

    # Evaluating Test set
    prediction = classifier.predict(X_test)

    # Evaluating Result
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    cnf_matrix = confusion_matrix(Y_test, prediction)

    # Accuracy score
    accuracy = accuracy_score(Y_test, prediction)

    # precision score (When is 0 and should be 1 and the other way round)
    precision = precision_score(Y_test, prediction)

    # recall score
    recall = recall_score(Y_test, prediction)
    score = f1_score(Y_test, prediction)

    # Metrics
    results = pd.DataFrame([[format(i), accuracy, precision, recall, score]],
                           columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 score'])
    print(results)
