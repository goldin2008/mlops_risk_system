import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import load_data
from diagnostics import model_predictions
from sklearn import metrics
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])



##############Function for reporting
def score_model(filename):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    file_path = os.path.join(test_data_path, filename)
    df_data, X, y = load_data(file_path)
    y_pred = model_predictions(X)
    logger.info(f"y_pred: {y_pred}")
    cm = metrics.confusion_matrix(y, y_pred)

    _ = sns.heatmap(cm)
    plt.title(f'Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # write the confusion matrix to the workspace
    fig = os.path.join(output_model_path, 'confusionmatrix.png')
    plt.savefig(fig)
    
    return score



if __name__ == '__main__':
    score = score_model()
