# Importing required dependencies 
import pandas as pd
from sklearn.svm import LinearSVC 
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from joblib import dump

# training a sentiment analysis model using TfidfVectorizer to transform the sentences 
# and applying LinearSVC() SVM model which is one of the most popular ML models 
# for this task

def train():

  train_path = "./train.csv"
  train_data = pd.read_csv(train_path)

  y_train = train_data['sentiment']
  X_train = train_data.drop(['sentiment'], axis = 1).squeeze() # convert to Series to avoid datatype error

  vectorizer = TfidfVectorizer(stop_words = list(ENGLISH_STOP_WORDS))
  vectorizer.fit(X_train)
  X_train = vectorizer.transform(X_train)

  model = LinearSVC()
  model.fit(X_train, y_train)

  # saving the model
  dump(model, 'linearsvc.joblib')

if __name__ == '__main__':
    train()

