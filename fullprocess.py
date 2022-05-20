import json
import logging
import os
import subprocess
import pandas as pd

import training
import scoring
import deployment
import diagnostics
import reporting
from sklearn import metrics

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path  = os.path.join(config['prod_deployment_path'])


##################Check and read new data
#first, read ingestedfiles.txt
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

def check_new_data():
    """
    Checks that the data files are the sames as the one in the ingested file
    :return: bool True if new data is found, False if not
    """
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'r') as f:
        ingested_files = f.read()

    source_files = set(os.listdir(input_folder_path))
    diff = source_files.difference(ingested_files)
    return True if diff==0 else False


def check_model_drift():
    """
    Checks whether the model has drifted
    :return:
    """
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        latest_score = float(f.read())

    file_path = os.path.join(output_folder_path, 'finaldata.csv')
    df_data = pd.read_csv(file_path)
    df = df_data.copy().drop("corporation", axis=1)
    y = df["exited"]
    X = df.drop("exited", axis=1)

    y_pred = diagnostics.model_predictions(X)
    new_score = metrics.f1_score(y, y_pred)

    return latest_score < new_score


def main():
    if check_new_data():
        subprocess.run(['python', 'ingestion.py'], stdout=subprocess.PIPE)
        if check_model_drift():
            subprocess.run(['python', 'training.py'], stdout=subprocess.PIPE)
            subprocess.run(['python', 'deployment.py'], stdout=subprocess.PIPE)
            subprocess.run(['python', 'scoring.py'], stdout=subprocess.PIPE)
            subprocess.run(['python', 'diagnostics.py'], stdout=subprocess.PIPE)
            subprocess.run(['python', 'reporting.py'], stdout=subprocess.PIPE)

if __name__ == "__main__":
    main()
