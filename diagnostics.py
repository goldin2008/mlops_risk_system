
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import logging
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deploy_path = os.path.join(config['prod_deployment_path'])

##################Function to get data
def load_data(file_path):
    df_data = pd.read_csv(file_path)
    df = df_data.copy().drop("corporation", axis=1)
    y = df["exited"]
    X = df.drop("exited", axis=1)
    return df_data, X, y

##################Function to get model predictions
def model_predictions(X):
    #read the deployed model and a test dataset, calculate predictions
    model_path = os.path.join(prod_deploy_path, "trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    y_pred = model.predict(X)
    return y_pred #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary(df):
    #calculate summary statistics here
    numeric_data = df.drop(['corporation', 'exited'], axis=1)
    data_summary = numeric_data.agg(['mean', 'median', 'std'])
    return data_summary #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system(f"python ingestion.py")
    time_ingestion = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.system(f"python training.py")
    time_training = timeit.default_timer() - start_time
    return [time_ingestion, time_training] #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf-8')
    return str(outdated)    

##################Function to check missing data
def missing_data(df):
    na_per = (df.isna().sum() / df.shape[0] * 100)
    return na_per

if __name__ == '__main__':
    file_path = os.path.join(test_data_path, 'testdata.csv')
    df_data, X, y = load_data(file_path)
    y_pred = model_predictions(X)
    print(f"y_pred: {y_pred}")

    data_summary = dataframe_summary(df_data)
    print(f"data_summary: {data_summary}")
    
    na_per = missing_data(df_data)
    print(f"na_per: {na_per}")
    
    exe_time = execution_time()
    print(f"exe_time: {exe_time}")

    outdated = outdated_packages_list()
    print(f"outdated: {outdated}")
