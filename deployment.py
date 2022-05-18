from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copy2


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path'])


####################function for deployment
def deploy_model():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    file_dict = {
        'ingestedfiles.txt': dataset_csv_path,
        'trainedmodel.pkl': output_model_path,
        'latestscore.txt': output_model_path,
    }       

    if not os.path.exists(config["prod_deployment_path"]):
        os.makedirs(config["prod_deployment_path"])

    for file, path in file_dict.items():
        copy2(os.path.join(path, file), prod_deployment_path)
        
if __name__ == "__main__":
    deploy_model()
