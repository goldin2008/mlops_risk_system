import pandas as pd
import numpy as np
import os
import json
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
    """
    Merge multiple dataframes into one dataframe
    :return: None
    """
    #check for datasets, compile them together, and write to an output file
    col_names = ['corporation', 'lastmonth_activity', 'lastyear_activity',
                 'number_of_employees', 'exited']

    df_list = pd.DataFrame(columns=col_names)

    logger.info("reading and merging all found .csv files")

    # recursivly search direcotries and read .csv files.
    r_path = os.path.join(os.getcwd(), input_folder_path)

    datasets = glob.glob(f'{r_path}/**/*.csv', recursive=True)
    df_list = pd.concat(map(pd.read_csv, datasets))

    logger.info("clean data and write to outputfolder")

    final_data = df_list.drop_duplicates()
    final_data.to_csv(os.path.join(output_folder_path,
                      'finaldata.csv'), index=False)

    logger.info("extract and save consumed filenames")
    # FUTURE: consider also save the path and output to .json

    file_names = [os.path.basename(path) for path in datasets]


    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as f:
        for element in file_names:
            f.write(element + "\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
    logger.info("Data Ingestion Done\n")
