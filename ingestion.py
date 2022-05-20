import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

import logging
import glob


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df = pd.DataFrame()

    # recursivly search direcotries and read .csv files.
    datasets = glob.glob(f'{input_folder_path}/*.csv', recursive=True)
    df = pd.concat(map(pd.read_csv, datasets))

    df_final = df.drop_duplicates()
    df_final.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    file_list = [os.path.basename(filepath) for filepath in datasets]

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as f:
        for file in file_list:
            f.write(file + "\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
