"""
Flask API with:
- Prediction Endpoint
- Scoring Endpoint
- Summary Statistic Endpoint
- Diagnostics Endpoint
"""
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics
import scoring
import subprocess


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    file_path = os.path.join(test_data_path, filename)
    _, X, _ = diagnostics.load_data(file_path)
    y_pred = diagnostics.model_predictions(X)
    return str(y_pred) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    # score = subprocess.run(['python', 'scoring.py'], capture_output=True).stdout
    score = scoring.score_model()
    return str(score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    filename = request.args.get('filename')
    file_path = os.path.join(test_data_path, filename)
    df_data, _, _ = diagnostics.load_data(file_path)
    data_sum = diagnostics.dataframe_summary(df_data)
    return str(data_sum) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():        
    #check timing and percent NA values
    filename = request.args.get('filename')
    file_path = os.path.join(test_data_path, filename)
    df_data, _, _ = diagnostics.load_data(file_path)

    missing = diagnostics.missing_data(df_data)
    exe_time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()

    diag_res = "Missing Data: \n{} \n\nExe Time:\nIngestion:{} \nTraining:{} \n\nOutdated Packages:\n{}".format(missing, exe_time[0], exe_time[1], outdated)
    return diag_res #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
