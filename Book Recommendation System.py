#Making necesarry imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors



# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

#making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)


#load the data
file_path = 'datasets/'

#Loading data
books = pd.read_csv(file_path+'BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

users = pd.read_csv(file_path+'BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

ratings = pd.read_csv(file_path+'BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

#checking shapes of the datasets
print(books.shape);
print(users.shape);
print(ratings.shape);

#Exploring books dataset
#print(books.head());

#dropping last three columns containing image URLs which will not be required for analysis
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'],axis=1,inplace=True);

#Now the books datasets looks like....
print(books.head());

#checking data types of columns
print(books.dtypes);

#yearOfPublication should be set as having dtype as int checking the unique values of yearOfPublication
print(books.yearOfPublication.unique());

# As it can be seen from below that there are some incorrect entries in this field.
# It looks like Publisher names 'DK Publishing Inc' and 'Gallimard' have been incorrectly loaded as yearOfPublication
# in dataset due to some errors in csv file.
# Also some of the entries are strings and same years have been entered as numbers in some places



#investigating the rows having 'DK Publishing Inc' as yearOfPublication
#print(books.loc[books.yearOfPublication == 'DK Publishing Inc']);

#From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

#rechecking
print(books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X')]);
#corrections done

#investigating the rows having 'Gallimard' as yearOfPublication
#print(books.loc[books.yearOfPublication == 'Gallimard',:])

#making required corrections as above, keeping other fields intact
books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"

#rechecking
#print(books.loc[books.ISBN == '2070426769',:]);
#corrections done

#Correcting the dtypes of yearOfPublication

#If error is set to ‘raise’, then invalid parsing will raise an exception
#-> If ‘coerce’, then invalid parsing will be set as NaN
#-> If ‘ignore’, then invalid parsing will return the input
books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce')

print(sorted(books['yearOfPublication'].unique()));

#However, the value 0 is invalid and as this dataset was published in 2004, I have assumed the the years after 2006 to be
#invalid keeping some margin in case dataset was updated thereafer setting invalid years as NaN
books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN

#replacing NaNs with mean value of yearOfPublication
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)

#rechecking
print(books.yearOfPublication.isnull().sum());
#No NaNs

#resetting the dtype as int32
books.yearOfPublication = books.yearOfPublication.astype(np.int32)

#exploring 'publisher' column
print(books.loc[books.publisher.isnull(),:]);
#two NaNs

#investigating rows having NaNs
#Checking with rows having bookTitle as Tyrant Moon to see if we can get any clues
print(books.loc[(books.bookTitle == 'Tyrant Moon'),:]);
#no clues

#Checking with rows having bookTitle as Finder Keepers to see if we can get any clues
print(books.loc[(books.bookTitle == 'Finders Keepers'),:]);
#all rows with different publisher and bookAuthor

#checking by bookAuthor to find patterns
books.loc[(books.bookAuthor == 'Elaine Corvidae'),:]
#all having different publisher...no clues here

#checking by bookAuthor to find patterns
books.loc[(books.bookAuthor == 'Linnea Sinclair'),:]

#since there is nothing in common to infer publisher for NaNs, replacing these with 'other
books.loc[(books.ISBN == '193169656X'),'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'),'publisher'] = 'other'

print('-------------------- investigating user dataset -----------------');
print(users.shape);
print(users.head());

print(users.dtypes);

#print user id
print(users.userID.values);

#print age
print(sorted(users.Age.unique()));

#In my view values below 5 and above 90 do not make much sense for our book rating case. Hence replacing these by NaNs
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan

#replacing NaNs with mean
users.Age = users.Age.fillna(users.Age.mean())

#setting the data type as int
users.Age = users.Age.astype(np.int32)

#rechecking
print(sorted(users.Age.unique()));
#looks good now


print('-------------------- investing Ratings Dataset ----------------');
#checking shape
print(ratings.shape);

#ratings dataset will have n_users*n_books entries if every user rated every item, this shows that the dataset is very sparse
n_users = users.shape[0]
n_books = books.shape[0]
print(n_users * n_books);

#checking first few rows...
print(ratings.head(5))

print(ratings.bookRating.unique());

#ratings dataset should have books only which exist in our books dataset, unless new books are added to books dataset
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
print(ratings_new);

print(ratings.shape);
#it can be seen that many rows having book ISBN not part of books dataset got dropped off
print(ratings_new.shape);

#ratings dataset should have ratings from users which exist in users dataset, unless new users are added to users dataset
ratings_new = ratings[ratings.userID.isin(users.userID)]

print(ratings.shape)
#no new users added, hence we will go with above dataset ratings_new (1031136, 3)
print(ratings_new.shape)

print("number of users: " + str(n_users));
print("number of books: " + str(n_books));

#Sparsity of dataset in %
sparsity=1.0-len(ratings_new)/float(n_users*n_books)
print('The sparsity level of Book Crossing dataset is ' +  str(sparsity*100) + ' %')

#As quoted in the description of the dataset - BX-Book-Ratings contains the book rating information.
#Ratings are either explicit, expressed on a scale from 1-10 higher values denoting higher appreciation, or implicit, expressed by 0
print(ratings.bookRating.unique());

#Hence segragating implicit and explict ratings datasets
ratings_explicit = ratings_new[ratings_new.bookRating != 0]
ratings_implicit = ratings_new[ratings_new.bookRating == 0]

#checking shapes
print(ratings_new.shape);
print(ratings_explicit.shape);
print(ratings_implicit.shape);

#plotting count of bookRating
sns.countplot(data=ratings_explicit , x='bookRating')
#It can be seen that higher ratings are more common amongst users and rating 8 has been rated highest number of times
#plt.show()

#At this point , a simple popularity based recommendation system can be built based on count of user ratings for different books
ratings_count = pd.DataFrame(ratings_explicit.groupby(['ISBN'])['bookRating'].sum())
top10 = ratings_count.sort_values('bookRating', ascending = False).head(10)
print("Following books are recommended");
top10.merge(books, left_index = True, right_on = 'ISBN')
print('------------------------------');
print(top10);

#Similarly segregating users who have given explicit ratings from 1-10 and those whose implicit behavior was tracked
users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]

#checking shapes
print(users.shape)
print(users_exp_ratings.shape)
print(users_imp_ratings.shape)

#To cope up with computing power I have and to reduce the dataset size,
# I am considering users who have rated atleast 100 books and books which have atleast 100 ratings
counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]
counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 100].index)]

#Generating ratings matrix from explicit ratings table
ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating')
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
print(ratings_matrix.shape)
ratings_matrix.head()

n_users = ratings_matrix.shape[0] #considering only those users who gave explicit ratings
n_books = ratings_matrix.shape[1]
print(n_users, n_books);

#since NaNs cannot be handled by training algorithms, replacing these by 0, which indicates absence of ratings
#setting data type
ratings_matrix.fillna(0, inplace = True)
ratings_matrix = ratings_matrix.astype(np.int32)
#checking first few rows
print(ratings_matrix.head(5));

#rechecking the sparsity
sparsity=1.0-len(ratings_explicit)/float(users_exp_ratings.shape[0]*n_books)
print('The sparsity level of Book Crossing dataset is ' +  str(sparsity*100) + ' %');

#--------------------------------------------
# setting global variables
global metric, k
k = 10
metric = 'cosine'

print(' ################### User-based Recommendation System ########################')

# This function finds k similar users given the user_id and ratings matrix
# These similarities are same as obtained via using pairwise_distances
def findksimilarusers(user_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()

    return similarities, indices


# This function predicts rating for specified user-item combination based on user-based approach
def predict_userbased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = 0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices = findksimilarusers(user_id, ratings, metric, k)  # similar users based on cosine similarity
    mean_rating = ratings.iloc[user_loc, :].mean()  # to adjust for zero based indexing
    sum_wt = np.sum(similarities) - 1
    product = 1
    wtd_sum = 0

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue;
        else:
            ratings_diff = ratings.iloc[indices.flatten()[i], item_loc] - np.mean(ratings.iloc[indices.flatten()[i], :])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product

    # in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    # which are handled here as below
    if prediction <= 0:
        prediction = 1
    elif prediction > 10:
        prediction = 10

    prediction = int(round(mean_rating + (wtd_sum / sum_wt)))
    print
    '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id, item_id, prediction)

    return prediction

print(predict_userbased(11676,'0001056107',ratings_matrix));


print(' ################### Item-based Recommendation Systems ########################')

# This function finds k similar items given the item_id and ratings matrix
def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    ratings = ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()

    return similarities, indices


similarities, indices = findksimilaritems('0001056107', ratings_matrix)

# This function predicts the rating for specified user-item combination based on item-based approach
def predict_itembased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = wtd_sum = 0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices = findksimilaritems(item_id, ratings)  # similar users based on correlation coefficients
    sum_wt = np.sum(similarities) - 1
    product = 1
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue;
        else:
            product = ratings.iloc[user_loc, indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum / sum_wt))

    # in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    # which are handled here as below //code has been validated without the code snippet below, below snippet is to avoid negative
    # predictions which might arise in case of very sparse datasets when using correlation metric
    if prediction <= 0:
        prediction = 1
    elif prediction > 10:
        prediction = 10

    print
    '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id, item_id, prediction)

    return prediction


prediction = predict_itembased(11676, '0001056107', ratings_matrix)
print(prediction)
