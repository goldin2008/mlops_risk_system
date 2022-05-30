
import timeit
import os
import sys
import json
import pickle
import logging
import subprocess
import pandas as pd
import numpy as np

from config import DATA_PATH, TEST_DATA_PATH, PROD_DEPLOYMENT_PATH

logging.basicConfig(stream=sys.stdout,level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

##################Load config.json and get environment variables


##################Function to get data
def load_data():
    logger.info("Load Data")
    file_path = os.path.join(TEST_DATA_PATH, 'testdata.csv')
    df_data = pd.read_csv(file_path)
    df = df_data.copy().drop("corporation", axis=1)
    y = df["exited"]
    X = df.drop("exited", axis=1)
    return df_data, X, y

##################Function to get model predictions
def model_predictions():
    logger.info("Model Predictions")
    _, X, _ = load_data()
    #read the deployed model and a test dataset, calculate predictions
    model_path = os.path.join(PROD_DEPLOYMENT_PATH, "trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    y_pred = model.predict(X)
    return y_pred #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    logger.info("Dataframe Summary")
    df, _, _ = load_data()
    #calculate summary statistics here
    # numeric_data = df.drop(['corporation', 'exited'], axis=1)
    # data_summary = numeric_data.agg(['mean', 'median', 'std'])

    data_df = df.drop(['exited'], axis=1)
    data_df = df.select_dtypes('number')
    logging.info("Calculating statistics for data")
    data_summary = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()
        data_summary[col] = {'mean': mean, 'median': median, 'std': std}

    return data_summary #return value should be a list containing all summary statistics

##################Function to check missing data
def missing_data():
    """
    Calculates percentage of missing data for each column
    in finaldata.csv
    Returns:
        list[dict]: Each dict contains column name and percentage
    """
    logger.info("Loading and preparing finaldata.csv")
    df, _, _ = load_data()
    # data_df = data_df.drop(['corporation', 'exited'], axis=1)

    logging.info("Calculating missing data percentage")
    missing_list = {col: {'percentage': perc} for col, perc in zip(
        df.columns, df.isna().sum() / df.shape[0] * 100)}

    return missing_list

##################Function to get timings
def _ingestion_timing():
    """
    Runs ingestion.py script and measures execution time

    Returns:
        float: running time
    """
    start_time = timeit.default_timer()
    os.system(f"python ingestion.py")
    # _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - start_time
    return timing

def _training_timing():
    """
    Runs training.py script and measures execution time

    Returns:
        float: running time
    """
    start_time = timeit.default_timer()
    os.system(f"python training.py")
    # _ = subprocess.run(['python', 'training.py'], capture_output=True)
    timing = timeit.default_timer() - start_time
    return timing

def execution_time():
    logger.info("Execution Time")
    #calculate timing of training.py and ingestion.py
    logger.info("Calculating time for ingestion.py")
    ingestion_time = []
    for _ in range(5):
        time = _ingestion_timing()
        ingestion_time.append(time)

    logging.info("Calculating time for training.py")
    training_time = []
    for _ in range(5):
        time = _training_timing()
        training_time.append(time)

    ret_list = [
        {'ingest_time_mean': np.mean(ingestion_time)},
        {'train_time_mean': np.mean(training_time)}
    ]

    return ret_list #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    # outdated = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf-8')

    logger.info("Checking outdated dependencies")
    dependencies = subprocess.run('pip-outdated ./requirements.txt',
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  shell=True,
                                  encoding='utf-8')
    # print('dependencies: ', dependencies)
    dep = dependencies.stdout
    # print('dep: ', dep)
    dep = dep.translate(str.maketrans('', '', ' \t\r'))
    dep = dep.split('\n')
    dep = [dep[3]] + dep[5:-3]
    dep = [s.split('|')[1:-1] for s in dep]

    return dep

if __name__ == '__main__':
    # y_pred = model_predictions()
    # print(f"y_pred: {y_pred}")

    # data_summary = dataframe_summary()
    # print(f"data_summary: {data_summary}")
    
    # na_per = missing_data()
    # print(f"na_per: {na_per}")
    
    # exe_time = execution_time()
    # print(f"exe_time: {exe_time}")

    # outdated = outdated_packages_list()
    # print(f"outdated: {outdated}")

    print("Model predictions on testdata.csv:",
        model_predictions(), end='\n\n')

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end='\n\n')

    print("Missing percentage")
    print(json.dumps(missing_data(), indent=4), end='\n\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n\n')

    print("Outdated Packages")
    dependencies = outdated_packages_list()
    for row in dependencies:
        print('{:<20}{:<10}{:<10}{:<10}'.format(*row))
