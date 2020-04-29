import pandas as pd

dataset = pd.read_csv('./datasets/Social_Network_Ads.csv')
print(dataset.head())

# Considering Age and EstimatedSalary col only
X = dataset.iloc[:, [2, 3]].values

# Considering Purchased col only
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
Here since data type of the feature are different hence we need to perform Feature Scaling.
"This is to standardize the range of independent variables or features of data"
If this is not done, result would be affected
'''

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs' )
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)