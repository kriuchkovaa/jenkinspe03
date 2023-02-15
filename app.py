# importing required dependencies 
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Define a flask app 
app = Flask(__name__)

@app.route("/")
def home():
  html = "<h2>Sklearn Prediction Model Root</h2>"
  return html

# Run the test 
@app.route('/inference', methods = ['POST'])
def inference():
  
  filename = request.json['filename'] #this is another issue
  
  test_data = pd.read_csv(filename)
  
  # Note that train.csv is needed here to run the vectorizer successfully 
  # (cannot perform fitting without it!)
  train_path = './train.csv'
  train_data = pd.read_csv(train_path)

  y_test = test_data['sentiment']
  X_test = test_data.drop(['sentiment'], axis = 1).squeeze()  
  X_train = train_data.drop(['sentiment'], axis = 1).squeeze()

  # Note that squeeze() option in the split performed above is needed to convert sets to 
  # Series; otherwise, difference in size and error 

  # Note: it turns out that in Docker frozenset (datatype of ENGLISH_STOP_WORDS) 
  # causes an error - thus, need to convert to a list
  vectorizer = TfidfVectorizer(stop_words = list(ENGLISH_STOP_WORDS))
  vectorizer.fit(X_train)
  X_test = vectorizer.transform(X_test)

  # load the training model and display results to the user 
  linear_svc = joblib.load('linearsvc.joblib')
  print("Classification result:")
  print(linear_svc.score(X_test, y_test))
  print(linear_svc.predict(X_test))
  result = str(linear_svc.score(X_test))
  return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)