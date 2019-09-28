# aml_test.py: AML testing script for sklearn
# Author: Nishanth Koganti
# Date: 2019/09/28

# import modules
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# aml imports
import azureml
from azureml.core.model import Model
from azureml.core import Workspace, Run
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies 

# display the core SDK version number
print('Azure ML SDK Version: ', azureml.core.VERSION)

# initialize aml workspace
ws = Workspace.from_config('.azureml/config.json')

model = Model(ws, 'sklearn_classification')
model.download(target_dir=os.getcwd(), exist_ok=True)

# create webservice config file
aciconfig = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1, tags={'data': 'mnist', 'method': 'sklearn'},
    description='Predict mnist with sklearn')

# setup inference configuration
inference_folder = os.path.join(
    os.getcwd(), 'sklearn_classification', 'inference')

inference_config = InferenceConfig(
    source_directory=inference_folder, runtime='python', 
    entry_script='score.py', conda_file='deploy_env.yml')

# deploy model service
service = Model.deploy(
    workspace=ws, name='sklearn-classification', models=[model],
    inference_config=inference_config, deployment_config=aciconfig)

# obtain scoring uri
service.wait_for_deployment(show_output=True)
print(service.scoring_uri)
