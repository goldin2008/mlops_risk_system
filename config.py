import os
import json
import yaml

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# output_folder_path = os.path.join(config['output_folder_path']) 
# test_data_path = os.path.join(config['test_data_path']) 
# prod_deployment_path = os.path.join(config['prod_deployment_path'])

# INPUT_FOLDER_PATH = os.path.join(os.path.abspath('../'),
#                                  'data',
#                                  config['input_folder_path'])
# DATA_PATH = os.path.join(os.path.abspath('../'),
#                          'data',
#                          config['output_folder_path'])
# TEST_DATA_PATH = os.path.join(os.path.abspath('../'),
#                               'data',
#                               config['test_data_path'])
# OUTPUT_MODEL_PATH = os.path.join(os.path.abspath('../'),
#                           'model',
#                           config['output_model_path'])
# PROD_DEPLOYMENT_PATH = os.path.join(os.path.abspath('../'),
#                                     'model',
#                                     config['prod_deployment_path'])

INPUT_FOLDER_PATH = os.path.join(os.path.abspath('./'),
                                 config['input_folder_path'])
DATA_PATH = os.path.join(os.path.abspath('./'),
                         config['output_folder_path'])
TEST_DATA_PATH = os.path.join(os.path.abspath('./'),
                              config['test_data_path'])
OUTPUT_MODEL_PATH = os.path.join(os.path.abspath('./'),
                          config['output_model_path'])
PROD_DEPLOYMENT_PATH = os.path.join(os.path.abspath('./'),
                                    config['prod_deployment_path'])