from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB
# scikit packages
from sklearn.preprocessing import StandardScaler

# plot functions
# import plot_functions as pf
# import pf as pf

# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

#making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)

#Load Data
df = pd.read_csv("./datasets/creditcard.csv")
#print(df.head())

# determine the number of records in the dataset
print('The dataset contains {0} rows and {1} columns.'.format(df.shape[0], df.shape[1]))
#print(df.columns)
#print(df.info())
#print(df.describe())

#Explore label class
print('Transactions count: \n', df['Class'].value_counts())
#print('Normal transactions count: \n', df['Class'].value_counts().values[0])
#print('Fraudulent transactions count: ', df['Class'].value_counts().values[1])

#Separate feature data (predictors) from labels
# feature data (predictors)
X = df.iloc[:, :-1]
# label class
Y = df['Class']

#Standardize data : Scale the data to have zero mean and unit variance.
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Partition data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size=0.33, random_state=42)

# apply the ADASYN over-sampling
ada = ADASYN(random_state=42)
print('Original dataset shape {}'.format(Counter(Y_train)))
print('Original dataset shape {}'.format(Y_train.value_counts()))
X_res, Y_res = ada.fit_sample(X_train, Y_train)
print('Resampled dataset shape {}'.format(Counter(Y_res)))

#Train Models
'''
Three machine learning algorithms: 
    Logistic Regression, 
    Naive Baye, and 
    RandomForest classifiers 
were trained using the processed feature data.
'''
X_train, Y_train = X_res, Y_res

# Train LogisticRegression Model
LGR_Classifier = LogisticRegression(solver='lbfgs')
LGR_Classifier.fit(X_train, Y_train)

# Train Decision Tree Model
RDF_Classifier = RandomForestClassifier(random_state=0)
RDF_Classifier.fit(X_train, Y_train)

# Train Bernoulli Naive Baye Model
BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, Y_train)

# Evaluate models
modelsList = [('RandomForest Classifier', RDF_Classifier),('LogisticRegression', LGR_Classifier),
('Naive Baiye Classifier', BNB_Classifier)]

models = [j for j in modelsList]

print()
print('========================== Model Evaluation Results ========================' "\n")

for i, v in models:
    scores = cross_val_score(v, X_train, Y_train, cv=10)
    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
    classification = metrics.classification_report(Y_train, v.predict(X_train))

    print('===== {} ====='.format(i))
    print()
    print ("Cross Validation Mean Score: ", '{}%'.format(np.round(scores.mean(), 3) * 100))
    print()
    print ("Model Accuracy: ", '{}%'.format(np.round(accuracy, 3) * 100))
    print()
    print("Confusion Matrix:" "\n", confusion_matrix)
    print()
    print("Classification Report:" "\n", classification)
    print()

# Test models
classdict = {'normal':0, 'fraudulent':1}
print()
print('========================== Model Test Results ========================' "\n")   

for i, v in models:
    accuracy = metrics.accuracy_score(Y_test, v.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(X_test))
    classification = metrics.classification_report(Y_test, v.predict(X_test))
    print('=== {} ==='.format(i))
    print ("Model Accuracy: ",  '{}%'.format(np.round(accuracy, 3) * 100))
    print()
    print("Confusion Matrix:" "\n", confusion_matrix)
    print()
    
    #pf.plot_confusion_matrix(confusion_matrix, classes = list(classdict.keys()), title='Confusion Matrix Plot', cmap=plt.cm.summer)
    print() 
    print("Classification Report:" "\n", classification) 
    print() 

print('============================= ROC Curve ===============================' "\n")      
#pf.plot_roc_auc(arg1=models, arg2=X_test, arg3=Y_test)
