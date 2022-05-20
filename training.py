from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data  
    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    # encode corporation code to numeric value
    df['corporation'] = df['corporation'].apply(lambda x: sum(bytearray(x, 'utf-8')))
    df = df.drop("corporation", axis=1)
    y = df["exited"]
    X = df.drop("exited", axis=1)
    model.fit(X, y)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.exists(config["output_model_path"]):
        os.makedirs(config["output_model_path"])

    model_output_path = os.path.join(
        config["output_model_path"], "trainedmodel.pkl"
    )

    pickle.dump(model, open(model_output_path, "wb"))

if __name__ == "__main__":
    train_model()