# import modules
import os
import argparse
import requests
import numpy as np
import urllib.request

# relative imports
from sklearn_classification.training.utils import load_data

# argument parser
parser = argparse.ArgumentParser(
    description='test inference script')
parser.add_argument('--uri', type=str, required=True,
                    help='inference endpoint uri')
args = parser.parse_args()

# download test dataset
data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

urllib.request.urlretrieve(
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    filename=os.path.join(data_folder, 'train-images.gz'))
urllib.request.urlretrieve(
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    filename=os.path.join(data_folder, 'train-labels.gz'))
urllib.request.urlretrieve(
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    filename=os.path.join(data_folder, 'test-images.gz'))
urllib.request.urlretrieve(
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    filename=os.path.join(data_folder, 'test-labels.gz'))

# load test dataset
X_test = load_data(os.path.join(
    data_folder, 'test-images.gz'), False) / 255.0
y_test = load_data(os.path.join(
    data_folder, 'test-labels.gz'), True).reshape(-1)

# send a random row from the test set to score
random_index = np.random.randint(0, len(X_test)-1)
input_data = '{\"data\": [' + str(list(X_test[random_index])) + ']}'

# setup scoring uri
scoring_uri = args.uri

# post a request
headers = {'Content-Type': 'application/json'}
resp = requests.post(scoring_uri, input_data, headers=headers)

print('post to url ', scoring_uri)

print('prediction: ', resp.text)
print('label: ', y_test[random_index])