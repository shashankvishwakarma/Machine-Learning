import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

#making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)

#Load Data
df1 = pd.read_csv("./datasets/delhi-weather-data.csv")
df = df1.iloc[:5000]
#print(df.head())

# determine the number of records in the dataset
print('The dataset contains {0} rows and {1} columns.'.format(df.shape[0], df.shape[1]))
#print(df.shape)
#print(df.columns)
#print(df.describe())

# We can see all the column name has space in there names, lets assign new names with removed space.
#print(df.columns)
df.columns = map(lambda x: x.strip(), df.columns)
#print(df.columns)

# Lets check the usual whether condition. We can see usually delhi's weather is either Haze, Smoke.
print(df._conds.value_counts(ascending=False))

# Lets plot top 10 weather condition in delhi.
plt.figure(figsize=(15, 10))
df._conds.value_counts().head(10).plot(kind='bar')
plt.title("Top 10 most common weather condition")
plt.plot()
#plt.show()

# Lets see top 10 least condition
plt.figure(figsize=(15, 10))
df._conds.value_counts(ascending=True).head(10).plot(kind="bar")
plt.title("Top 10 least whether condition in delhi")
plt.plot()
#plt.show()

# common wind direction
print("============common wind direction============")
print(df._wdire.value_counts())

# Average temperature
print("average temperature in delhi:", round(df._tempm.mean(axis=0),2))

# As we can see there is datetime column, We can extract year from it.
# Year can ve an important feature for us to calculate how temperature is changing
# according to year function to get year
def extract_year(value):
    return (value[0:4])

# function to get month
def extract_month(value):
    return (value[4:6])

# Lets check our method
df["year"] = df["datetime_utc"].apply(lambda x:extract_year(x))
df["month"] = df["datetime_utc"].apply(lambda x:extract_month(x))

# lets check out data range # So our given data is from 1996 to 2017 as per this example
print("max [{}], min[{}]: ".format(df.year.max(),df.year.min()))

# Number of records for particular year
print('=========== Number of records for particular year ============')
print(df.year.value_counts())

df.groupby("year")._tempm.mean()
df_mean = df.groupby("year")._tempm.mean().reset_index().sort_values('_tempm', ascending=True)
print(df_mean.dtypes)

# Changing data type for year to float
df_mean.year = df_mean.year.astype("float")
print(df_mean.dtypes)

# Missing Values
print('========== Missing Values ===============')
print(df.isnull().sum())

# We will make copy of original dataset and will take only relevant columns.
df_filtered = df[['datetime_utc', '_conds', '_dewptm', '_fog', '_hail',
       '_hum', '_pressurem', '_rain', '_snow', '_tempm', '_thunder',
       '_tornado', '_vism', '_wdird', '_wdire', '_wspdm', 'year', "month"]]

# Lets replace missing values in _dewptm. We can take an average of that year
print(df_filtered[df_filtered._dewptm.isnull()].head(5))

# if you see pressure column, there are few -9999 values.
# Which is obviously bad values and it can affect your calculations very badly.
# So we will consider this also missing values. Lets convert them first to the nan
df_filtered._pressurem.replace(-9999.0, np.nan, inplace=True)

# We will try to replace value with average value of that year
def replace_with_average(col_name):
    for index, row in df_filtered[df_filtered[col_name].isnull()].iterrows():
        mean_val = df_filtered[df_filtered["year"] == row["year"]][col_name].mean()
        df_filtered.at[index, col_name] = mean_val

null_columns_list1 = ['_dewptm','_hum','_pressurem','_tempm','_vism','_wdird','_wspdm']
for col in null_columns_list1:
    replace_with_average(col)

# As we can see _wdire is a categorical feature so we can not apply mean here. We have to get the most frequent
# value of _wdire for a year and then replace missing value with the most frequent value.
def replace_with_max(col_name):
    for index, row in df_filtered[df_filtered[col_name].isnull()].iterrows():
        most_frequent = df_filtered[df_filtered["year"] == row["year"]][col_name].value_counts().idxmax()
        df_filtered.at[index, col_name] = most_frequent

null_columns_list2 = ['_wdire','_conds']
for col in null_columns_list2:
    replace_with_average(col)

# Missing Values on filtered col, WE HAVE REPLACED ALL THE MISSING VALUES.
print('========== Missing Values on filtered col ===============')
print(df_filtered.isnull().sum())

df_filtered.year = df_filtered.year.astype("object")
df_filtered.month = df_filtered.month.astype("object")

# Heatmap for year and average temprature across the month. More red more heat, more blue less heat
pd.crosstab(df_filtered.year, [df_filtered.month], values=df_filtered._tempm, aggfunc="mean")
plt.figure(figsize=(15, 10))
sns.heatmap(pd.crosstab(df_filtered.year, [df_filtered.month], values=df_filtered._tempm, aggfunc="mean"),
            cmap="coolwarm", annot=True, cbar=True)
plt.title("Average Temprature 1996-2016")
plt.plot()
#plt.show()

'''
Now our dataset doesn;t have any missing values in it. Now we should observe one thing. 
That our _windre is a categorical column and it is also important to predict a whether 
but the thing is your model does not understand a text value. 
So we need to encode this categorical column so that we can change it to integer
'''
print(df_filtered._conds.value_counts())

# Feature & Target Matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

feature_columns = ['_wdire', '_dewptm', '_fog', '_hail', '_hum',
       '_pressurem', '_rain', '_snow', '_tempm', '_thunder', '_tornado',
       '_vism', '_wdird', '_wspdm', 'year', 'month', '_conds']

# Lets create a new dataset, so that we dont change in our filtered dataset
# We will create dataset in such a way, _wdire(categorical feature in starting position
# & target variable at last which is _conds
df_final = df_filtered[feature_columns]

# This will convert '_wdire' values column and for value it will add 1 for that row
# or else 0
wdire_dummies = pd.get_dummies(df_final["_wdire"])
df_final = pd.concat([wdire_dummies, df_final], axis=1)
df_final.drop("_wdire", inplace=True, axis=1)

X = df_final.iloc[:, 0:-1].values
y = df_final.iloc[:, -1].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train & Test Split
from sklearn.model_selection import train_test_split

# test size =0.25 or 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
print("Shape of X_train", X_train.shape)
print("Shape of X_test", X_test.shape)
print("Shape of y_train", y_train.shape)
print("Shape of y_test", y_test.shape)

# Create Model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=0)

# Train Model
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)

# Accuracy
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, prediction))