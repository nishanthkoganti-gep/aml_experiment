# import modules
import os
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from IPython.terminal.debugger import set_trace as keyboard

# aml imports
import azureml.core
from azureml.core import Run
from azureml.core import Workspace
from azureml.core import Experiment

# check core sdk version number
print('Azure ML SDK Version: ', azureml.core.VERSION)

# load workspace instance
ws = Workspace.from_config(path='.azureml/config.json')
print(ws.name, ws.location, ws.resource_group, sep='\t')

# create classification experiment
experiment_name = 'sklearn-classification'
exp = Experiment(workspace=ws, name=experiment_name)

# obtain run with max accuracy
max_accuracy = None
max_acc_runid = None

for run in exp.get_runs():
    run_metrics = run.get_metrics()
    run_details = run.get_details()
 
    # obtain metric and run id
    run_id = run_details['runId']
    run_acc = run_metrics['accuracy']

    if max_accuracy is None:
        max_accuracy = run_acc
        max_acc_runid = run_id
    else:
        if run_acc > max_accuracy:
            max_accuracy = run_acc
            max_acc_runid = run_id

# obtain best run
print("Best run_id: " + max_acc_runid)
print("Best run_id rmse: " + str(max_accuracy))
best_run = Run(experiment=exp, run_id=max_acc_runid)

# register model for deployment
model = best_run.register_model(
    model_name='sklearn_classification',
    model_path='outputs/sklearn_classification_model.pkl')
print(model.name, model.id, model.version, sep='\t')
