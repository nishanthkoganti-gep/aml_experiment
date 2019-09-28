# aml_train.py: AML training script for sklearn
# Author: Nishanth Koganti
# Date: 2019/09/28

# import modules
import os
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

# aml imports
import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.dataset import Dataset
from azureml.train.sklearn import SKLearn
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# check core sdk version number
print('Azure ML SDK Version: ', azureml.core.VERSION)

# load workspace instance
ws = Workspace.from_config(path='.azureml/config.json')
print(ws.name, ws.location, ws.resource_group, sep='\t')

# create classification experiment
experiment_name = 'sklearn-classification'
exp = Experiment(workspace=ws, name=experiment_name)

# choose a name for your cluster and node limits
compute_min_nodes = os.environ.get('AML_COMPUTE_CLUSTER_MIN_NODES', 0)
compute_max_nodes = os.environ.get('AML_COMPUTE_CLUSTER_MAX_NODES', 4)
compute_name = os.environ.get('AML_COMPUTE_CLUSTER_NAME', 'sklearn-classify')

# setup cpu based vm for computation
vm_size = os.environ.get('AML_COMPUTE_CLUSTER_SKU', 'STANDARD_D2_V2')

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found compute target. Using: ' + compute_name)
else:
    print('Create compute target. Using: ' + compute_name)

    # setup provisioning configuration
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                min_nodes=compute_min_nodes,
                                                                max_nodes=compute_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(
        ws, compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout.
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)

    # for a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())

# register dataset to be used in compute
web_paths = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
            ]
dataset = Dataset.File.from_files(path=web_paths)

dataset = dataset.register(workspace=ws,
                           name='classification dataset',
                           description='training and test dataset',
                           create_new_version=True)

# create directory for training scripts
train_folder = os.path.join(
    os.getcwd(), 'sklearn_classification', 'training')

# create environment for classification
env = Environment('classification_env')
cd = CondaDependencies.create(pip_packages=['azureml-sdk', 'scikit-learn',
                                            'azureml-dataprep[pandas,fuse]>=1.1.14'])
env.python.conda_dependencies = cd

# setup hyper parameter values to tune
regularizations = np.linspace(0.05,0.95,10)

# loop over the parameter values
for reg in regularizations:
    # create sklearn estimator
    train_params = {
        '--data-folder': dataset.as_named_input('data').as_mount(),
        '--regularization': reg
    }

    est = SKLearn(
        source_directory=train_folder, script_params=train_params,
        compute_target=compute_target, environment_definition=env, 
        entry_script='train.py')

    # submit run for execution
    run = exp.submit(config=est)
