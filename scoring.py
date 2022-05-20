from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Take trained model
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    y_test = test_df.pop('exited')
    X_test = test_df.drop(['corporation'], axis=1)

    y_pred = model.predict(X_test)
    score = metrics.f1_score(y_test, y_pred)

    score_path = os.path.join(output_model_path, "latestscore.txt")
    with open(score_path, 'w') as f:
        f.write(str(score))

    return score

if __name__ == "__main__":
    score = score_model()
