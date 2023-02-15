import pandas as pd
import joblib
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

def inference():
  
  test_path = "./test.csv"
  test_data = pd.read_csv(test_path)
  
  train_path = './train.csv'
  train_data = pd.read_csv(train_path)

  y_test = test_data['sentiment']
  X_test = test_data.drop(['sentiment'], axis = 1).squeeze()
  X_train = train_data.drop(['sentiment'], axis = 1).squeeze()

  vectorizer = TfidfVectorizer(stop_words = list(ENGLISH_STOP_WORDS))
  vectorizer.fit(X_train)
  X_test = vectorizer.transform(X_test)

  linear_svc = joblib.load('linearsvc.joblib')
  #print("Classification result:")
  #print(linear_svc.score(X_test, y_test))
  #print(linear_svc.predict(X_test))
  result = linear_svc.score(X_test, y_test)
  return result

if __name__ == '__main__':
    inference()