import os
import requests
import json
import subprocess

#Specify a URL that resolves to your workspace
# URL = "http://127.0.0.1/"
URL = "http://127.0.0.1:8000/"

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

output_model_path = os.path.join(config['output_model_path'])

#Call each API endpoint and store the responses
#### Request module solution####
response1 = requests.get(f'{URL}/prediction?filename=testdata.csv').content #put an API call here
response2 = requests.get(f'{URL}/scoring').content #put an API call here
response3 = requests.get(f'{URL}/summarystats?filename=testdata.csv').content #put an API call here
response4 = requests.get(f'{URL}/diagnostics?filename=testdata.csv').content #put an API call here

#### Command-line solution####
response5=subprocess.run(['curl', '127.0.0.1:8000/prediction?filename=testdata.csv'],capture_output=True).stdout
response6=subprocess.run(['curl', '127.0.0.1:8000/scoring'],capture_output=True).stdout
response7=subprocess.run(['curl', '127.0.0.1:8000/summarystats?filename=testdata.csv'],capture_output=True).stdout
response8=subprocess.run(['curl', '127.0.0.1:8000/diagnostics?filename=testdata.csv'],capture_output=True).stdout

#combine all API responses
responses = (
    f"Perdiction:\n {response1}\n"
    f"Scoring:\n  {response2}\n"
    f"Summary Stats:\n {response3}\n"
    f"Diagnostics:\n {response4}") #combine reponses here

#write the responses to your workspace
with open(os.path.join(output_model_path, "apireturns.txt"), 'w') as output_file:
    output_file.writelines(responses)
