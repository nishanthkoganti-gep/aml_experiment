# import modules
import os
import json
import pickle
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

# aml imports
from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = os.path.join(
        os.getenv('AZUREML_MODEL_DIR'), 'sklearn_mnist_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    
    # make prediction
    y_hat = model.predict(data)
    
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()