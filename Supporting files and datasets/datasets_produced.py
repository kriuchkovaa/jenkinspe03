# Importing required dependencies
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the original dataset (additional attributes added to avoid a tokenization error caused by empty lines)
data = pd.read_csv('IMDB Dataset.csv', engine = 'python', error_bad_lines = False)

# Note: Link to a dataset - https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/data 

# Checking the file content
data.head()

# Getting rid of hashtags in the body of comments
data['review'] = data['review'].str.replace('(<br />|\d+\.)','').str.split().agg(" ".join)

# Removing the punctuation with RegEx command
data['review'] = data['review'].str.replace(r'[^\w\s]+', ' ')

# Converting all sentences in 'review' column to a lowercase
data['review'] = data['review'].apply(str.lower)

# Changing labels in 'sentiment' column to integer counterparts
data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})

# Assigning values to X and y
X = data['review']
y = data['sentiment']

# Splitting the data in 70 to 30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

# Ensuring the same datatype by converting the subsets into dataframes
X_train = X_train.to_frame()
X_test = X_test.to_frame()
y_train = y_train.to_frame()
y_test = y_test.to_frame()

# Creating train_data and test_data and loading the output into a csv file 
# index = False allows to get rid of an additional index column in each file 
train_data = pd.concat([X_train, y_train], axis="columns").to_csv('train.csv', index = False)
test_data= pd.concat([X_test, y_test], axis = 'columns').to_csv('test.csv', index = False)

